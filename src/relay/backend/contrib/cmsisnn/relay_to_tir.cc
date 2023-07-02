/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ir/transform.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"
#include "buffer_size.h"
#include "compiler_attrs.h"
#include "convolutions.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*!
 * \brief This is a helper class to generate tir.Buffers and BufferMap
 *
 * The PrimFuncs generated needs Buffers produced to attach information
 * about the inputs and output tir::Vars. This is helper class to generate
 * them in the relay to TIR lowering
 */
class BufferCreator {
 public:
  /*! \brief Creates a tir::Var and tir::Buffer then returns the buffer var to be used by the body
   */
  tir::Var CreateBufferVar(String name_hint, DataType dtype) {
    tir::Var var = tir::Var(name_hint, dtype);
    tir::Buffer buffer = tir::decl_buffer({}, DataType::Int(dtype.bits()), name_hint + "_");
    _primfunc_params_.push_back(var);
    _buffer_map_.Set(var, buffer);
    _buffer_vars_.Set(name_hint, buffer->data);
    return buffer->data;
  }
  /*! \brief Access already created buffer_var by associated tir::Var name */
  tir::Var GetBufferVar(String name_hint) { return _buffer_vars_[name_hint]; }
  /*! \brief Get the BufferMap that maps tir::Var to tir::Buffer */
  Map<tir::Var, tir::Buffer> GetBufferMap() { return _buffer_map_; }
  /*! \brief Get the PrimFunc params that is a collection of tir::Vars created in the process */
  Array<tir::Var> GetPrimFuncParams() { return _primfunc_params_; }

 private:
  Map<String, tir::Var> _buffer_vars_;
  Map<tir::Var, tir::Buffer> _buffer_map_;
  Array<tir::Var> _primfunc_params_;
};

class RelayToTIRVisitor : public MixedModeMutator {
 public:
  explicit RelayToTIRVisitor(IRModule ir_module, Target target)
      : ir_module_(ir_module), target_(target) {
    context_buffer_id_ = 0;
  }

  IRModule Mutate() {
    GlobalVar main_global_var = ir_module_->GetGlobalVar("main");
    Function main = Downcast<Function>(ir_module_->Lookup(main_global_var));
    Function mutated_main = WithFields(main, main->params, VisitExpr(main->body));

    ir_module_->Update(main_global_var, mutated_main);

    return ir_module_;
  }

 private:
  inline IntImm ToArg(int32_t value) { return IntImm(DataType::Int(32), value); }

  void CreatePrimFuncForExtern(const GlobalVar& global_var, Array<tir::Var> func_signature,
                               const Map<tir::Var, tir::Buffer>& buffer_map,
                               tvm::Array<PrimExpr> call_extern_args,
                               PrimExpr context_buffer_var = PrimExpr(),
                               int context_buffer_size = 0, int num_bits = 8) {
    Map<String, ObjectRef> dict_attrs;
    dict_attrs.Set(tvm::attr::kGlobalSymbol, global_var->name_hint);
    dict_attrs.Set(tvm::attr::kTarget, target_);
    dict_attrs.Set("tir.noalias", Bool(true));

    tir::Stmt body = tir::Evaluate(
        tvm::tir::Call(DataType::Int(num_bits), tir::builtin::call_extern(), call_extern_args));

    if (context_buffer_size) {
      body = tir::Allocate(Downcast<tir::Var>(context_buffer_var), DataType::Int(num_bits),
                           {context_buffer_size}, tir::const_true(), body);
    }

    tir::PrimFunc replacement_func(func_signature, body, VoidType(), buffer_map,
                                   DictAttrs(dict_attrs));
    ir_module_->Add(global_var, replacement_func);
  }

  auto GetIntMinMax(int bit_width) {
    const int32_t min =
        (bit_width == 8) ? std::numeric_limits<int8_t>::min() : std::numeric_limits<int16_t>::min();
    const int32_t max =
        (bit_width == 8) ? std::numeric_limits<int8_t>::max() : std::numeric_limits<int16_t>::max();
    return std::pair(min, max);
  }

  auto GetClipMinMax(const ClipAttrs* clip_attrs) {
    return std::pair(static_cast<int32_t>(clip_attrs->a_min),
                     static_cast<int32_t>(clip_attrs->a_max));
  }

  auto GetClipMinMax(const Call& clip_op) { return GetClipMinMax(clip_op->attrs.as<ClipAttrs>()); }

  void Fiti_MAMM(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* add_call = expr.as<CallNode>();
    const CallNode* mul_call = add_call->args[0].as<CallNode>();

    Array<PrimExpr> input_shape = add_call->type_as<TensorTypeNode>()->shape;

    const float add_input_scale = GetScalarFromConstant<float>(add_call->args[2]);
    const int32_t add_input_zp = GetScalarFromConstant<int32_t>(add_call->args[3]);
    const float add_const_scale = GetScalarFromConstant<float>(add_call->args[4]);
    const int32_t add_const_zp = GetScalarFromConstant<int32_t>(add_call->args[5]);
    float add_output_scale = GetScalarFromConstant<float>(add_call->args[6]);
    int32_t add_output_zp = GetScalarFromConstant<int32_t>(add_call->args[7]);

    const float mul_input_scale = GetScalarFromConstant<float>(mul_call->args[2]);
    const int32_t mul_input_zp = GetScalarFromConstant<int32_t>(mul_call->args[3]);
    const float mul_const_scale = GetScalarFromConstant<float>(mul_call->args[4]);
    const int32_t mul_const_zp = GetScalarFromConstant<int32_t>(mul_call->args[5]);
    float mul_output_scale = GetScalarFromConstant<float>(mul_call->args[6]);
    int32_t mul_output_zp = GetScalarFromConstant<int32_t>(mul_call->args[7]);

    const float max_input_scale = std::max(add_input_scale, add_const_scale);
    const double twice_max_input_scale = 2 * static_cast<double>(max_input_scale);
    const double scaled_add_input_scale =
        static_cast<double>(add_input_scale) / twice_max_input_scale;
    const double scaled_add_const_scale =
        static_cast<double>(add_const_scale) / twice_max_input_scale;

    double add_scale = twice_max_input_scale / ((1 << 20) * static_cast<double>(add_output_scale));
    double mul_scale = static_cast<double>(mul_input_scale) * static_cast<double>(mul_const_scale) /
                       static_cast<double>(mul_output_scale);

    auto add_in_mult_shift_pair =
        tvm::relay::qnn::GetFixedPointMultiplierShift(scaled_add_input_scale);
    int32_t add_in_multiplier = std::get<0>(add_in_mult_shift_pair);
    int32_t add_in_shift = std::get<1>(add_in_mult_shift_pair) + 20;
    auto add_const_mult_shift_pair =
        tvm::relay::qnn::GetFixedPointMultiplierShift(scaled_add_const_scale);
    int32_t add_const_multiplier = std::get<0>(add_const_mult_shift_pair);
    int32_t add_const_shift = std::get<1>(add_const_mult_shift_pair) + 20;
    auto add_out_mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(add_scale);
    int32_t add_out_multiplier = std::get<0>(add_out_mult_shift_pair);
    int32_t add_out_shift = std::get<1>(add_out_mult_shift_pair);
    auto mul_mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(mul_scale);
    int32_t mul_multiplier = std::get<0>(mul_mult_shift_pair);
    int32_t mul_shift = std::get<1>(mul_mult_shift_pair);

    PrimExpr tensor_size = mul_call->type_as<TensorTypeNode>()->Size();

    BufferCreator buffer_creator;
    tir::Var input = buffer_creator.CreateBufferVar("input", DataType::Handle(8));
    tir::Var Const_mul = buffer_creator.CreateBufferVar("Const_mul", DataType::Handle(8));
    tir::Var Const_add = buffer_creator.CreateBufferVar("Const_add", DataType::Handle(8));
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(8));

    tvm::Array<PrimExpr> args = {
        tir::StringImm("fiti_mamm"),
        input,
        Const_mul,
        Const_add,
        output,
        ToArg(-mul_input_zp),
        ToArg(-mul_const_zp),
        ToArg(mul_output_zp),
        ToArg(mul_multiplier),
        ToArg(mul_shift),
        ToArg(-add_input_zp),
        ToArg(add_in_multiplier),
        ToArg(add_in_shift),
        ToArg(-add_const_zp),
        ToArg(add_const_multiplier),
        ToArg(add_const_shift),
        ToArg(add_output_zp),
        ToArg(add_out_multiplier),
        ToArg(add_out_shift),
        tensor_size,
    };

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  void Fiti_Concat(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* concat_call = expr.as<CallNode>();
    const ConcatenateAttrs* concat_attrs = concat_call->attrs.as<ConcatenateAttrs>();
    auto concat_input = concat_call->args[0]->type_as<TupleTypeNode>();
    int input_num = concat_input->fields.size();
    auto axis = concat_attrs->axis;

    if (axis == -1) axis = concat_input->fields[0].as<TensorTypeNode>()->shape.size() - 1;

    int input_args[4][2];
    for (int i = 0; i < input_num; i++) {
      int tmp = 1;
      for (int j = 0; j < axis; j++)
        tmp *= qnn::get_const_int(concat_input->fields[i].as<TensorTypeNode>()->shape[j]);
      input_args[i][0] = tmp;
      tmp = 1;
      for (long unsigned int j = axis; j < concat_input->fields[i].as<TensorTypeNode>()->shape.size(); j++)
        tmp *= qnn::get_const_int(concat_input->fields[i].as<TensorTypeNode>()->shape[j]);
      input_args[i][1] = tmp;
    }

    BufferCreator buffer_creator;
    tir::Var input1 = buffer_creator.CreateBufferVar("input1", DataType::Handle(8));
    tir::Var input2 = buffer_creator.CreateBufferVar("input2", DataType::Handle(8));

    tvm::Array<PrimExpr> args;
    if (input_num == 3) {
      tir::Var input3 = buffer_creator.CreateBufferVar("input3", DataType::Handle(8));
      tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(8));
      args = {
          tir::StringImm("fiti_concat"),
          input1,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[0][0]),
          IntImm(DataType::Int(32), input_args[0][1]),
          input2,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[1][0]),
          IntImm(DataType::Int(32), input_args[1][1]),
          input3,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[2][0]),
          IntImm(DataType::Int(32), input_args[2][1]),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          output,
          IntImm(DataType::Int(32), input_num),
      };
    } else if (input_num == 4) {
      tir::Var input3 = buffer_creator.CreateBufferVar("input3", DataType::Handle(8));
      tir::Var input4 = buffer_creator.CreateBufferVar("input4", DataType::Handle(8));
      tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(8));
      args = {
          tir::StringImm("fiti_concat"),
          input1,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[0][0]),
          IntImm(DataType::Int(32), input_args[0][1]),
          input2,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[1][0]),
          IntImm(DataType::Int(32), input_args[1][1]),
          input3,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[2][0]),
          IntImm(DataType::Int(32), input_args[2][1]),
          input4,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[3][0]),
          IntImm(DataType::Int(32), input_args[3][1]),
          output,
          IntImm(DataType::Int(32), input_num),
      };
    } else {
      tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(8));
      args = {
          tir::StringImm("fiti_concat"),
          input1,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[0][0]),
          IntImm(DataType::Int(32), input_args[0][1]),
          input2,
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), 1),
          IntImm(DataType::Int(32), input_args[1][0]),
          IntImm(DataType::Int(32), input_args[1][1]),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          IntImm(DataType::Int(32), 0),
          output,
          IntImm(DataType::Int(32), input_num),
      };
    }

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  void Fiti_Slice(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* reshape_call = expr.as<CallNode>();
    const CallNode* slice_call = reshape_call->args[0].as<CallNode>();
    const StridedSliceAttrs* slice_attrs = slice_call->attrs.as<StridedSliceAttrs>();

    Array<PrimExpr> input_shape = slice_call->args[0]->type_as<TensorTypeNode>()->shape;
    auto begin = slice_attrs->begin.value()[3];
    auto end = slice_attrs->end.value()[3];

    BufferCreator buffer_creator;
    tir::Var input = buffer_creator.CreateBufferVar("input", DataType::Handle(8));
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(8));

    tvm::Array<PrimExpr> args = {
        tir::StringImm("fiti_slice"),
        input,
        output,
        ToArg(qnn::get_const_int(begin)),
        ToArg(qnn::get_const_int(end)),
        ToArg(qnn::get_const_int(input_shape[1])),
        ToArg(qnn::get_const_int(input_shape[2])),
        ToArg(qnn::get_const_int(input_shape[3])),
    };
    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  void Fiti_Requant(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* requant_call = expr.as<CallNode>();
    Array<PrimExpr> input_shape = requant_call->type_as<TensorTypeNode>()->shape;
    int32_t input_size = 1;
    for (size_t i = 1; i < input_shape.size(); i++)
      input_size *= qnn::get_const_int(input_shape[i]);
    int32_t in_offset = GetScalarFromConstant<int32_t>(requant_call->args[2]);
    int32_t out_offset = GetScalarFromConstant<int32_t>(requant_call->args[4]);
    float in_scale = GetScalarFromConstant<float>(requant_call->args[1]);
    float out_scale = GetScalarFromConstant<float>(requant_call->args[3]);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(
        static_cast<double>(in_scale) / static_cast<double>(out_scale));
    int32_t multiplier = std::get<0>(mult_shift_pair);
    int32_t shift = std::get<1>(mult_shift_pair);

    BufferCreator buffer_creator;
    tir::Var input = buffer_creator.CreateBufferVar("input", DataType::Handle(8));
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(8));

    tvm::Array<PrimExpr> args = {
        tir::StringImm("fiti_requant"),        input,
        IntImm(DataType::Int(32), input_size), IntImm(DataType::Int(32), -in_offset),
        IntImm(DataType::Int(32), out_offset), IntImm(DataType::Int(32), -128),
        IntImm(DataType::Int(32), 127),        IntImm(DataType::Int(32), multiplier),
        IntImm(DataType::Int(32), shift),      output,
    };
    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  void Fiti_Resize2d(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* resize2d_call = expr.as<CallNode>();
    const VarNode* input_Var = resize2d_call->args[0].as<VarNode>();
    const Resize2DAttrs* resize2d_attrs = resize2d_call->attrs.as<Resize2DAttrs>();
    Array<PrimExpr> input_shape = input_Var->type_as<TensorTypeNode>()->shape;

    BufferCreator buffer_creator;
    tir::Var input = buffer_creator.CreateBufferVar("input", DataType::Handle(8));
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(8));

    tvm::Array<PrimExpr> args = {
        tir::StringImm("fiti_resize2d"),
        input,
        IntImm(DataType::Int(32), qnn::get_const_int(input_shape[1])),
        IntImm(DataType::Int(32), qnn::get_const_int(input_shape[3])),
        IntImm(DataType::Int(32), qnn::get_const_int(resize2d_attrs->size[0])),
        output,
    };
    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  void Fiti_Sigmoid(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* reshape_call = NULL;
    const CallNode* quant_call = NULL;
    if (expr.as<CallNode>()->op.as<OpNode>()->name == "reshape") {
      reshape_call = expr.as<CallNode>();
      quant_call = reshape_call->args[0].as<CallNode>();
    } else
      quant_call = expr.as<CallNode>();

    const CallNode* sigmoid_call = quant_call->args[0].as<CallNode>();
    const CallNode* dequant_call = sigmoid_call->args[0].as<CallNode>();
    float input_scale = GetScalarFromConstant<float>(dequant_call->args[1]);
    int32_t input_zero_point = GetScalarFromConstant<int32_t>(dequant_call->args[2]);
    float output_scale = GetScalarFromConstant<float>(quant_call->args[1]);
    int32_t output_zero_point = GetScalarFromConstant<int32_t>(quant_call->args[2]);

    Array<PrimExpr> input_shape = dequant_call->type_as<TensorTypeNode>()->shape;
    int32_t input_size = qnn::get_const_int(input_shape[1]) * qnn::get_const_int(input_shape[2]) *
                         qnn::get_const_int(input_shape[3]);

    BufferCreator buffer_creator;
    tir::Var input = buffer_creator.CreateBufferVar("input", DataType::Handle(8));
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(8));

    tvm::Array<PrimExpr> args = {
        tir::StringImm("fiti_sigmoid"),
        input,
        ToArg(input_size),
        FloatImm(DataType::Float(32), input_scale),
        ToArg(-input_zero_point),
        FloatImm(DataType::Float(32), output_scale),
        ToArg(output_zero_point),
        output,
    };
    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  void fiti_reduce_transaction1(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* clip_call1 = nullptr;
    const CallNode* requantize_call1 = nullptr;
    const CallNode* bias_add_call1 = nullptr;
    const CallNode* conv2d_call1 = nullptr;
    const CallNode* clip_call2 = nullptr;
    const CallNode* requantize_call2 = nullptr;
    const CallNode* bias_add_call2 = nullptr;
    const CallNode* conv2d_call2 = nullptr;
    const CallNode* clip_call3 = nullptr;
    const CallNode* requantize_call3 = nullptr;
    const CallNode* bias_add_call3 = nullptr;
    const CallNode* conv2d_call3 = nullptr;
    const CallNode* final_call = expr.as<CallNode>();
    const OpNode* final_op = final_call->op.as<OpNode>();

    if (final_op->name == "clip") {
      clip_call3 = final_call;
      requantize_call3 = clip_call3->args[0].as<CallNode>();
    } else if (final_op->name == "qnn.requantize") {
      if (final_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "clip") {
        clip_call3 = final_call->args[0].as<CallNode>();
        requantize_call3 = clip_call3->args[0].as<CallNode>();
      } else
        requantize_call3 = final_call;
    }

    const CallNode* requantize3_input = requantize_call3->args[0].as<CallNode>();
    const OpNode* requantize3_input_op = requantize3_input->op.as<OpNode>();
    if (requantize3_input_op->name == "nn.bias_add") {
      bias_add_call3 = requantize3_input;
      conv2d_call3 = bias_add_call3->args[0].as<CallNode>();
    } else
      conv2d_call3 = requantize3_input;

    if ((conv2d_call3->args[0]).as<CallNode>()->op.as<OpNode>()->name == "clip") {
      clip_call2 = (conv2d_call3->args[0]).as<CallNode>();
      requantize_call2 = (clip_call2->args[0]).as<CallNode>();
    } else
      requantize_call2 = (conv2d_call3->args[0]).as<CallNode>();

    const CallNode* requantize2_input = (requantize_call2->args[0]).as<CallNode>();
    const OpNode* requantize2_input_op = requantize2_input->op.as<OpNode>();
    if (requantize2_input_op->name == "nn.bias_add") {
      bias_add_call2 = requantize2_input;
      conv2d_call2 = (bias_add_call2->args[0]).as<CallNode>();
    } else
      conv2d_call2 = requantize2_input;

    if ((conv2d_call2->args[0]).as<CallNode>()->op.as<OpNode>()->name == "clip") {
      clip_call1 = (conv2d_call2->args[0]).as<CallNode>();
      requantize_call1 = (clip_call1->args[0]).as<CallNode>();
    } else
      requantize_call1 = (conv2d_call2->args[0]).as<CallNode>();

    const CallNode* requantize1_input = (requantize_call1->args[0]).as<CallNode>();
    const OpNode* requantize1_input_op = requantize1_input->op.as<OpNode>();
    if (requantize1_input_op->name == "nn.bias_add") {
      bias_add_call1 = requantize1_input;
      conv2d_call1 = (bias_add_call1->args[0]).as<CallNode>();
    } else
      conv2d_call1 = requantize1_input;

    const int filter_scale_pos = 3;
    const int input_scale_pos = bias_add_call2 ? 5 : 4;
    BufferCreator buffer_creator;

    tir::Var input1 = buffer_creator.CreateBufferVar("input1", DataType::Handle(8));
    tir::Var filter1 = buffer_creator.CreateBufferVar("filter1", DataType::Handle(8));
    tir::Var multiplier1 = buffer_creator.CreateBufferVar("multiplier1", DataType::Handle(32));
    tir::Var bias1 = buffer_creator.CreateBufferVar("bias1", DataType::Handle(32));
    tir::Var shift1 = buffer_creator.CreateBufferVar("shift1", DataType::Handle(32));
    // tir::Var output1 = buffer_creator.CreateBufferVar("output1", DataType::Handle(8));

    // tir::Var input2 = buffer_creator.CreateBufferVar("input2", DataType::Handle(8));
    tir::Var filter2 = buffer_creator.CreateBufferVar("filter2", DataType::Handle(8));
    tir::Var multiplier2 = buffer_creator.CreateBufferVar("multiplier2", DataType::Handle(32));
    tir::Var bias2 = buffer_creator.CreateBufferVar("bias2", DataType::Handle(32));
    tir::Var shift2 = buffer_creator.CreateBufferVar("shift2", DataType::Handle(32));
    // tir::Var output2 = buffer_creator.CreateBufferVar("output2", DataType::Handle(8));

    // tir::Var input3 = buffer_creator.CreateBufferVar("input3", DataType::Handle(8));
    tir::Var filter3 = buffer_creator.CreateBufferVar("filter3", DataType::Handle(8));
    tir::Var multiplier3 = buffer_creator.CreateBufferVar("multiplier3", DataType::Handle(32));
    tir::Var bias3 = buffer_creator.CreateBufferVar("bias3", DataType::Handle(32));
    tir::Var shift3 = buffer_creator.CreateBufferVar("shift3", DataType::Handle(32));
    tir::Var output3 = buffer_creator.CreateBufferVar("output3", DataType::Handle(8));

    // Relay function contains input_scale and filter_scale as function parameters at the following
    // locations in the global partitioned function for Conv2D
    skip_call_args_.insert(filter_scale_pos);
    skip_call_args_.insert(input_scale_pos);
    skip_call_args_.insert(filter_scale_pos + 6);
    skip_call_args_.insert(input_scale_pos + 6);
    skip_call_args_.insert(filter_scale_pos + 6 * 2);
    skip_call_args_.insert(input_scale_pos + 6 * 2);
    // Individual arguments to the structs arguments of the CMSIS-NN API are filled into call_extern
    // https://github.com/ARM-software/CMSIS_5/blob/def6f800f95661eb3451d317f7d0dde504f6020d/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c#L50

    // prepare cmsis_nn_conv_params
    const Conv2DAttrs* conv2d_attrs3 = conv2d_call3->attrs.as<Conv2DAttrs>();
    int32_t input_offset3 = -GetScalarFromConstant<int32_t>(conv2d_call3->args[2]);
    int32_t output_offset3 = GetScalarFromConstant<int32_t>(requantize_call3->args[4]);
    int32_t stride_w3 = qnn::get_const_int(conv2d_attrs3->strides[1]);
    int32_t stride_h3 = qnn::get_const_int(conv2d_attrs3->strides[0]);
    int32_t padding_w3 = qnn::get_const_int(conv2d_attrs3->padding[1]);
    int32_t padding_h3 = qnn::get_const_int(conv2d_attrs3->padding[0]);
    int32_t dilation_w3 = qnn::get_const_int(conv2d_attrs3->dilation[1]);
    int32_t dilation_h3 = qnn::get_const_int(conv2d_attrs3->dilation[0]);
    int32_t out_channels3 = qnn::get_const_int(conv2d_attrs3->channels);
    std::string kernel_layout3 = conv2d_attrs3->kernel_layout.c_str();
    int32_t clip_min3 = std::numeric_limits<int8_t>::min();
    int32_t clip_max3 = std::numeric_limits<int8_t>::max();
    if (clip_call3) {
      const ClipAttrs* clip_attrs3 = clip_call3->attrs.as<ClipAttrs>();
      clip_min3 = clip_attrs3->a_min;
      clip_max3 = clip_attrs3->a_max;
    }

    const Conv2DAttrs* conv2d_attrs2 = conv2d_call2->attrs.as<Conv2DAttrs>();
    int32_t input_offset2 = -GetScalarFromConstant<int32_t>(conv2d_call2->args[2]);
    int32_t output_offset2 = GetScalarFromConstant<int32_t>(requantize_call2->args[4]);
    int32_t stride_w2 = qnn::get_const_int(conv2d_attrs2->strides[1]);
    int32_t stride_h2 = qnn::get_const_int(conv2d_attrs2->strides[0]);
    int32_t padding_w2 = qnn::get_const_int(conv2d_attrs2->padding[1]);
    int32_t padding_h2 = qnn::get_const_int(conv2d_attrs2->padding[0]);
    int32_t dilation_w2 = qnn::get_const_int(conv2d_attrs2->dilation[1]);
    int32_t dilation_h2 = qnn::get_const_int(conv2d_attrs2->dilation[0]);
    int32_t out_channels2 = qnn::get_const_int(conv2d_attrs2->channels);
    std::string kernel_layout2 = conv2d_attrs2->kernel_layout.c_str();
    int32_t clip_min2 = std::numeric_limits<int8_t>::min();
    int32_t clip_max2 = std::numeric_limits<int8_t>::max();
    if (clip_call2) {
      const ClipAttrs* clip_attrs2 = clip_call2->attrs.as<ClipAttrs>();
      clip_min2 = clip_attrs2->a_min;
      clip_max2 = clip_attrs2->a_max;
    }

    const Conv2DAttrs* conv2d_attrs1 = conv2d_call1->attrs.as<Conv2DAttrs>();
    int32_t input_offset1 = -GetScalarFromConstant<int32_t>(conv2d_call1->args[2]);
    int32_t output_offset1 = GetScalarFromConstant<int32_t>(requantize_call1->args[4]);
    int32_t stride_w1 = qnn::get_const_int(conv2d_attrs1->strides[1]);
    int32_t stride_h1 = qnn::get_const_int(conv2d_attrs1->strides[0]);
    int32_t padding_w1 = qnn::get_const_int(conv2d_attrs1->padding[1]);
    int32_t padding_h1 = qnn::get_const_int(conv2d_attrs1->padding[0]);
    int32_t dilation_w1 = qnn::get_const_int(conv2d_attrs1->dilation[1]);
    int32_t dilation_h1 = qnn::get_const_int(conv2d_attrs1->dilation[0]);
    int32_t out_channels1 = qnn::get_const_int(conv2d_attrs1->channels);
    std::string kernel_layout1 = conv2d_attrs1->kernel_layout.c_str();
    int32_t clip_min1 = std::numeric_limits<int8_t>::min();
    int32_t clip_max1 = std::numeric_limits<int8_t>::max();
    if (clip_call1) {
      const ClipAttrs* clip_attrs1 = clip_call1->attrs.as<ClipAttrs>();
      clip_min1 = clip_attrs1->a_min;
      clip_max1 = clip_attrs1->a_max;
    }

    tvm::Array<PrimExpr> scalar_args = {ToArg(input_offset1),
                                        ToArg(output_offset1),
                                        ToArg(stride_w1),
                                        ToArg(stride_h1),
                                        ToArg(padding_w1),
                                        ToArg(padding_h1),
                                        ToArg(dilation_w1),
                                        ToArg(dilation_h1),
                                        ToArg(clip_min1),
                                        ToArg(clip_max1),
                                        ToArg(-1),
                                        ToArg(input_offset2),
                                        ToArg(output_offset2),
                                        ToArg(stride_w2),
                                        ToArg(stride_h2),
                                        ToArg(padding_w2),
                                        ToArg(padding_h2),
                                        ToArg(dilation_w2),
                                        ToArg(dilation_h2),
                                        ToArg(clip_min2),
                                        ToArg(clip_max2),
                                        ToArg(-1),
                                        ToArg(input_offset3),
                                        ToArg(output_offset3),
                                        ToArg(stride_w3),
                                        ToArg(stride_h3),
                                        ToArg(padding_w3),
                                        ToArg(padding_h3),
                                        ToArg(dilation_w3),
                                        ToArg(dilation_h3),
                                        ToArg(clip_min3),
                                        ToArg(clip_max3),
                                        ToArg(-1)};

    // CMSIS-NN data structure "cmsis_nn_dims" for ifm expects input layout as NHWC
    // This is the same layout we expect in Relay
    Array<PrimExpr> input_shape3 = conv2d_call3->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> filter_shape3 = conv2d_call3->args[1]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> bias_shape3{1, 1, 1, out_channels3};
    Array<PrimExpr> output_shape3 = conv2d_call3->type_as<TensorTypeNode>()->shape;

    Array<PrimExpr> input_shape2 = conv2d_call2->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> filter_shape2 = (conv2d_call2->args[1])->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> bias_shape2{1, 1, 1, out_channels2};
    Array<PrimExpr> output_shape2 = conv2d_call2->type_as<TensorTypeNode>()->shape;

    Array<PrimExpr> input_shape1 = conv2d_call1->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> filter_shape1 = conv2d_call1->args[1]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> bias_shape1{1, 1, 1, out_channels1};
    Array<PrimExpr> output_shape1 = conv2d_call1->type_as<TensorTypeNode>()->shape;

    std::string cmsisnn_api = "fiti_convolve_wrapper_s83";
    tvm::Array<PrimExpr> call_ext_args = {tir::StringImm(cmsisnn_api),
                                          input1,
                                          filter1,
                                          multiplier1,
                                          bias1,
                                          shift1,
                                          filter2,
                                          multiplier2,
                                          bias2,
                                          shift2,
                                          filter3,
                                          multiplier3,
                                          bias3,
                                          shift3,
                                          output3};

    scalar_args = tvm::runtime::Concat(scalar_args, input_shape1);
    scalar_args = tvm::runtime::Concat(scalar_args, filter_shape1);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape1);
    scalar_args = tvm::runtime::Concat(scalar_args, output_shape1);
    scalar_args = tvm::runtime::Concat(scalar_args, input_shape2);
    scalar_args = tvm::runtime::Concat(scalar_args, filter_shape2);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape2);
    scalar_args = tvm::runtime::Concat(scalar_args, output_shape2);
    scalar_args = tvm::runtime::Concat(scalar_args, input_shape3);
    scalar_args = tvm::runtime::Concat(scalar_args, filter_shape3);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape3);
    scalar_args = tvm::runtime::Concat(scalar_args, output_shape3);
    call_ext_args = tvm::runtime::Concat(call_ext_args, scalar_args);
    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), call_ext_args);
  }

  void fiti_reduce_transaction2(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* clip_call1 = nullptr;
    const CallNode* requantize_call1 = nullptr;
    const CallNode* bias_add_call1 = nullptr;
    const CallNode* conv2d_call1 = nullptr;
    const CallNode* pad_call1 = nullptr;
    const CallNode* clip_call2 = nullptr;
    const CallNode* requantize_call2 = nullptr;
    const CallNode* bias_add_call2 = nullptr;
    const CallNode* conv2d_call2 = nullptr;
    const CallNode* pad_call2 = nullptr;
    const CallNode* final_call = expr.as<CallNode>();
    const OpNode* final_op = final_call->op.as<OpNode>();

    if (final_op->name == "clip") {
      clip_call2 = final_call;
      requantize_call2 = clip_call2->args[0].as<CallNode>();
    } else {
      if (final_op->name == "qnn.requantize") {
        if (final_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "clip") {
          clip_call2 = final_call->args[0].as<CallNode>();
          requantize_call2 = clip_call2->args[0].as<CallNode>();
        } else
          requantize_call2 = final_call;
      } else
        requantize_call2 = final_call;
    }

    const CallNode* requantize2_input = requantize_call2->args[0].as<CallNode>();
    const OpNode* requantize2_input_op = requantize2_input->op.as<OpNode>();
    if (requantize2_input_op->name == "nn.bias_add") {
      bias_add_call2 = requantize2_input;
      conv2d_call2 = bias_add_call2->args[0].as<CallNode>();
    } else
      conv2d_call2 = requantize2_input;

    if ((conv2d_call2->args[0]).as<CallNode>()->op.as<OpNode>()->name == "clip") {
      clip_call1 = (conv2d_call2->args[0]).as<CallNode>();
      requantize_call1 = (clip_call1->args[0]).as<CallNode>();
    } else {
      if ((conv2d_call2->args[0]).as<CallNode>()->op.as<OpNode>()->name == "nn.pad") {
        pad_call2 = conv2d_call2->args[0].as<CallNode>();
        clip_call1 = pad_call2->args[0].as<CallNode>();
        requantize_call1 = clip_call1->args[0].as<CallNode>();
      }
    }

    const CallNode* requantize1_input = (requantize_call1->args[0]).as<CallNode>();
    const OpNode* requantize1_input_op = requantize1_input->op.as<OpNode>();
    if (requantize1_input_op->name == "nn.bias_add") {
      bias_add_call1 = requantize1_input;
      conv2d_call1 = (bias_add_call1->args[0]).as<CallNode>();
    } else
      conv2d_call1 = requantize1_input;

    if (conv2d_call1->args[0]->IsInstance<CallNode>() &&
        (conv2d_call1->args[0]).as<CallNode>()->op.as<OpNode>()->name == "nn.pad")
      pad_call1 = conv2d_call1->args[0].as<CallNode>();

    const int filter_scale_pos = 3;
    const int input_scale_pos = bias_add_call2 ? 5 : 4;
    BufferCreator buffer_creator;

    tir::Var input1 = buffer_creator.CreateBufferVar("input1", DataType::Handle(8));
    tir::Var filter1 = buffer_creator.CreateBufferVar("filter1", DataType::Handle(8));
    tir::Var multiplier1 = buffer_creator.CreateBufferVar("multiplier1", DataType::Handle(32));
    tir::Var bias1 = buffer_creator.CreateBufferVar("bias1", DataType::Handle(32));
    tir::Var shift1 = buffer_creator.CreateBufferVar("shift1", DataType::Handle(32));
    // tir::Var output1 = buffer_creator.CreateBufferVar("output1", DataType::Handle(8));

    // tir::Var input2 = buffer_creator.CreateBufferVar("input2", DataType::Handle(8));
    tir::Var filter2 = buffer_creator.CreateBufferVar("filter2", DataType::Handle(8));
    tir::Var multiplier2 = buffer_creator.CreateBufferVar("multiplier2", DataType::Handle(32));
    tir::Var bias2 = buffer_creator.CreateBufferVar("bias2", DataType::Handle(32));
    tir::Var shift2 = buffer_creator.CreateBufferVar("shift2", DataType::Handle(32));
    tir::Var output2 = buffer_creator.CreateBufferVar("output2", DataType::Handle(8));

    // Relay function contains input_scale and filter_scale as function parameters at the following
    // locations in the global partitioned function for Conv2D
    skip_call_args_.insert(filter_scale_pos);
    skip_call_args_.insert(input_scale_pos);
    skip_call_args_.insert(filter_scale_pos + 6);
    skip_call_args_.insert(input_scale_pos + 6);
    // Individual arguments to the structs arguments of the CMSIS-NN API are filled into call_extern
    // https://github.com/ARM-software/CMSIS_5/blob/def6f800f95661eb3451d317f7d0dde504f6020d/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c#L50

    // prepare cmsis_nn_conv_params
    const Conv2DAttrs* conv2d_attrs2 = conv2d_call2->attrs.as<Conv2DAttrs>();
    int32_t input_offset2 = -GetScalarFromConstant<int32_t>(conv2d_call2->args[2]);
    int32_t output_offset2 = GetScalarFromConstant<int32_t>(requantize_call2->args[4]);
    int32_t stride_w2 = qnn::get_const_int(conv2d_attrs2->strides[1]);
    int32_t stride_h2 = qnn::get_const_int(conv2d_attrs2->strides[0]);
    int32_t padding_w2 = qnn::get_const_int(conv2d_attrs2->padding[1]);
    int32_t padding_h2 = qnn::get_const_int(conv2d_attrs2->padding[0]);

    if (pad_call2) {
      const PadAttrs* pad_attrs2 = (conv2d_call2->args[0]).as<CallNode>()->attrs.as<PadAttrs>();
      padding_w2 = qnn::get_const_int(pad_attrs2->pad_width[2][0]);
      padding_h2 = qnn::get_const_int(pad_attrs2->pad_width[1][0]);
    }

    int32_t dilation_w2 = qnn::get_const_int(conv2d_attrs2->dilation[1]);
    int32_t dilation_h2 = qnn::get_const_int(conv2d_attrs2->dilation[0]);
    int32_t out_channels2 = qnn::get_const_int(conv2d_attrs2->channels);
    std::string kernel_layout2 = conv2d_attrs2->kernel_layout.c_str();
    int32_t clip_min2 = std::numeric_limits<int8_t>::min();
    int32_t clip_max2 = std::numeric_limits<int8_t>::max();
    if (clip_call2) {
      const ClipAttrs* clip_attrs2 = clip_call2->attrs.as<ClipAttrs>();
      clip_min2 = clip_attrs2->a_min;
      clip_max2 = clip_attrs2->a_max;
    }

    const Conv2DAttrs* conv2d_attrs1 = conv2d_call1->attrs.as<Conv2DAttrs>();
    int32_t input_offset1 = -GetScalarFromConstant<int32_t>(conv2d_call1->args[2]);
    int32_t output_offset1 = GetScalarFromConstant<int32_t>(requantize_call1->args[4]);
    int32_t stride_w1 = qnn::get_const_int(conv2d_attrs1->strides[1]);
    int32_t stride_h1 = qnn::get_const_int(conv2d_attrs1->strides[0]);
    int32_t padding_w1 = qnn::get_const_int(conv2d_attrs1->padding[1]);
    int32_t padding_h1 = qnn::get_const_int(conv2d_attrs1->padding[0]);

    if (pad_call1) {
      const PadAttrs* pad_attrs1 = (conv2d_call1->args[0]).as<CallNode>()->attrs.as<PadAttrs>();
      padding_w1 = qnn::get_const_int(pad_attrs1->pad_width[2][0]);
      padding_h1 = qnn::get_const_int(pad_attrs1->pad_width[1][0]);
    }

    int32_t dilation_w1 = qnn::get_const_int(conv2d_attrs1->dilation[1]);
    int32_t dilation_h1 = qnn::get_const_int(conv2d_attrs1->dilation[0]);
    int32_t out_channels1 = qnn::get_const_int(conv2d_attrs1->channels);
    std::string kernel_layout1 = conv2d_attrs1->kernel_layout.c_str();
    int32_t clip_min1 = std::numeric_limits<int8_t>::min();
    int32_t clip_max1 = std::numeric_limits<int8_t>::max();
    if (clip_call1) {
      const ClipAttrs* clip_attrs1 = clip_call1->attrs.as<ClipAttrs>();
      clip_min1 = clip_attrs1->a_min;
      clip_max1 = clip_attrs1->a_max;
    }

    tvm::Array<PrimExpr> scalar_args = {ToArg(input_offset1),
                                        ToArg(output_offset1),
                                        ToArg(stride_w1),
                                        ToArg(stride_h1),
                                        ToArg(padding_w1),
                                        ToArg(padding_h1),
                                        ToArg(dilation_w1),
                                        ToArg(dilation_h1),
                                        ToArg(clip_min1),
                                        ToArg(clip_max1),
                                        ToArg(-1),
                                        ToArg(input_offset2),
                                        ToArg(output_offset2),
                                        ToArg(stride_w2),
                                        ToArg(stride_h2),
                                        ToArg(padding_w2),
                                        ToArg(padding_h2),
                                        ToArg(dilation_w2),
                                        ToArg(dilation_h2),
                                        ToArg(clip_min2),
                                        ToArg(clip_max2),
                                        ToArg(-1)};

    // CMSIS-NN data structure "cmsis_nn_dims" for ifm expects input layout as NHWC
    // This is the same layout we expect in Relay
    Array<PrimExpr> input_shape2 = conv2d_call2->args[0]->type_as<TensorTypeNode>()->shape;
    if (pad_call2)
      input_shape2 =
          ((conv2d_call2->args[0]).as<CallNode>()->args[0])->type_as<TensorTypeNode>()->shape;

    // CMSIS-NN data structure "cmsis_nn_dims" for weights expects following layouts
    // OHWI for Conv2D and IHWO for Depthwise convolutions
    Array<PrimExpr> filter_shape2 = (conv2d_call2->args[1])->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> bias_shape2{1, 1, 1, out_channels2};
    Array<PrimExpr> output_shape2 = conv2d_call2->type_as<TensorTypeNode>()->shape;

    Array<PrimExpr> input_shape1 = conv2d_call1->args[0]->type_as<TensorTypeNode>()->shape;
    if (pad_call1)
      input_shape1 =
          ((conv2d_call1->args[0]).as<CallNode>()->args[0])->type_as<TensorTypeNode>()->shape;

    Array<PrimExpr> filter_shape1 = conv2d_call1->args[1]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> bias_shape1{1, 1, 1, out_channels1};
    Array<PrimExpr> output_shape1 = conv2d_call1->type_as<TensorTypeNode>()->shape;

    std::string cmsisnn_api = "fiti_convolve_wrapper_s82";

    tvm::Array<PrimExpr> call_ext_args = {tir::StringImm(cmsisnn_api),
                                          input1,
                                          filter1,
                                          multiplier1,
                                          bias1,
                                          shift1,
                                          filter2,
                                          multiplier2,
                                          bias2,
                                          shift2,
                                          output2};

    scalar_args = tvm::runtime::Concat(scalar_args, input_shape1);
    scalar_args = tvm::runtime::Concat(scalar_args, filter_shape1);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape1);
    scalar_args = tvm::runtime::Concat(scalar_args, output_shape1);
    scalar_args = tvm::runtime::Concat(scalar_args, input_shape2);
    scalar_args = tvm::runtime::Concat(scalar_args, filter_shape2);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape2);
    scalar_args = tvm::runtime::Concat(scalar_args, output_shape2);
    call_ext_args = tvm::runtime::Concat(call_ext_args, scalar_args);

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), call_ext_args);
  }

  void EmitConv2D(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* clip_call = nullptr;
    const CallNode* requantize_call = nullptr;
    const CallNode* bias_add_call = nullptr;
    const CallNode* conv2d_call = nullptr;
    const CallNode* final_call = expr.as<CallNode>();
    const OpNode* final_op = final_call->op.as<OpNode>();
    if (final_op->name == "clip") {
      clip_call = final_call;
      requantize_call = clip_call->args[0].as<CallNode>();
    } else {
      if (final_op->name == "qnn.requantize") {
        if (final_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "clip") {
          clip_call = final_call->args[0].as<CallNode>();
          requantize_call = clip_call->args[0].as<CallNode>();
        } else
          requantize_call = final_call;
      } else
        requantize_call = final_call;
    }
    const CallNode* requantize_input = requantize_call->args[0].as<CallNode>();
    const OpNode* requantize_input_op = requantize_input->op.as<OpNode>();
    if (requantize_input_op->name == "nn.bias_add") {
      bias_add_call = requantize_input;
      conv2d_call = bias_add_call->args[0].as<CallNode>();
    } else {
      conv2d_call = requantize_input;
    }
    int32_t dtype_bits = conv2d_call->args[0]->type_as<TensorTypeNode>()->dtype.bits();

    // Determine bitwidth of buffers based on input dtype
    int32_t input_bits = 8;
    int32_t filter_bits = 8;
    int32_t bias_bits = 32;
    int32_t output_bits = 8;
    int32_t context_buffer_bits = 8;
    bool is_int16 = false;
    if (dtype_bits == 16) {
      is_int16 = true;
      input_bits = 16;
      bias_bits = 64;
      output_bits = 16;
      context_buffer_bits = 16;
    }

    // TIR variables are created in the order they appear in the Relay partitioned function
    // %1 = qnn.conv2d(%input, %weight_const_0, input_zero_point_scalar,
    //                 %cmsisnn_multiplier_const_1, %input_scale_scalar, %weight_scale_const_2)
    // %2 = nn.bias_add(%1, %bias_const_3, axis=3)
    // %3 = qnn.requantize(%2, %input_scale_const_4, %cmsisnn_shift_const_5,
    //                     %output_scale_scalar, %output_zero_point_scalar)
    // clip(%3, a_min=%min_scalar, a_max=%max_scalar)
    // Position of scales in the global function for Conv2D
    const int filter_scale_pos = 3;
    const int input_scale_pos = bias_add_call ? 5 : 4;
    BufferCreator buffer_creator;
    tir::Var input = buffer_creator.CreateBufferVar("input", DataType::Handle(input_bits));
    tir::Var filter = buffer_creator.CreateBufferVar("filter", DataType::Handle(filter_bits));
    tir::Var multiplier = buffer_creator.CreateBufferVar("multiplier", DataType::Handle(32));
    if (bias_add_call) {
      buffer_creator.CreateBufferVar("bias", DataType::Handle(bias_bits));
    }
    tir::Var shift = buffer_creator.CreateBufferVar("shift", DataType::Handle(32));
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(output_bits));

    // Relay function contains input_scale and filter_scale as function parameters at the following
    // locations in the global partitioned function for Conv2D
    skip_call_args_.insert(filter_scale_pos);
    skip_call_args_.insert(input_scale_pos);

    // Individual arguments to the structs arguments of the CMSIS-NN API are filled into call_extern
    // https://github.com/ARM-software/CMSIS_5/blob/def6f800f95661eb3451d317f7d0dde504f6020d/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c#L50

    // prepare cmsis_nn_conv_params
    const Conv2DAttrs* conv2d_attrs = conv2d_call->attrs.as<Conv2DAttrs>();
    int32_t input_offset = -GetScalarFromConstant<int32_t>(conv2d_call->args[2]);
    int32_t output_offset = GetScalarFromConstant<int32_t>(requantize_call->args[4]);
    int32_t stride_w = qnn::get_const_int(conv2d_attrs->strides[1]);
    int32_t stride_h = qnn::get_const_int(conv2d_attrs->strides[0]);
    int32_t padding_w = qnn::get_const_int(conv2d_attrs->padding[1]);
    int32_t padding_h = qnn::get_const_int(conv2d_attrs->padding[0]);
    if (conv2d_call->args[0]->IsInstance<CallNode>() &&
        conv2d_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "nn.pad") {
      const PadAttrs* pad_attrs = conv2d_call->args[0].as<CallNode>()->attrs.as<PadAttrs>();
      padding_w = qnn::get_const_int(pad_attrs->pad_width[2][0]);
      padding_h = qnn::get_const_int(pad_attrs->pad_width[1][0]);
    }
    int32_t dilation_w = qnn::get_const_int(conv2d_attrs->dilation[1]);
    int32_t dilation_h = qnn::get_const_int(conv2d_attrs->dilation[0]);
    int32_t out_channels = qnn::get_const_int(conv2d_attrs->channels);
    std::string kernel_layout = conv2d_attrs->kernel_layout.c_str();
    const auto [clip_min, clip_max] =
        clip_call ? GetClipMinMax(GetRef<Call>(clip_call)) : GetIntMinMax(dtype_bits);

    tvm::Array<PrimExpr> scalar_args = {ToArg(input_offset), ToArg(output_offset), ToArg(stride_w),
                                        ToArg(stride_h),     ToArg(padding_w),     ToArg(padding_h),
                                        ToArg(dilation_w),   ToArg(dilation_h),    ToArg(clip_min),
                                        ToArg(clip_max)};

    // CMSIS-NN data structure "cmsis_nn_dims" for ifm expects input layout as NHWC
    // This is the same layout we expect in Relay
    Array<PrimExpr> input_shape = conv2d_call->args[0]->type_as<TensorTypeNode>()->shape;
    if (conv2d_call->args[0]->IsInstance<CallNode>() &&
        conv2d_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "nn.pad")
      input_shape = conv2d_call->args[0].as<CallNode>()->args[0]->type_as<TensorTypeNode>()->shape;
    int32_t input_n = qnn::get_const_int(input_shape[0]);
    int32_t input_h = qnn::get_const_int(input_shape[1]);
    int32_t input_c = qnn::get_const_int(input_shape[3]);

    // CMSIS-NN data structure "cmsis_nn_dims" for weights expects following layouts
    // OHWI for Conv2D and IHWO for Depthwise convolutions
    Array<PrimExpr> filter_shape = conv2d_call->args[1]->type_as<TensorTypeNode>()->shape;

    Array<PrimExpr> bias_shape{1, 1, 1, out_channels};

    Array<PrimExpr> output_shape = conv2d_call->type_as<TensorTypeNode>()->shape;
    int32_t output_h = qnn::get_const_int(output_shape[1]);
    int32_t output_w = qnn::get_const_int(output_shape[2]);
    int32_t output_c = qnn::get_const_int(output_shape[3]);

    int32_t depth_multiplier = -1;
    if (IsCMSISNNDepthwise(conv2d_attrs, input_shape, filter_shape)) {
      // Refer to TVM frontend to know how depth multiplier and out_channels are related
      // https://github.com/apache/tvm/blob/6ed3ab3e33f8eafa4acaf53b7a671831de7587e9/python/tvm/relay/frontend/tflite.py#L2129
      int kernel_pos_i = kernel_layout.find("I");
      int kernel_pos_o = kernel_layout.find("O");
      int kernel_pos_dm = input_c == 1 ? kernel_pos_o : kernel_pos_i;
      depth_multiplier = qnn::get_const_int(filter_shape[kernel_pos_dm]);
    }
    scalar_args.push_back(ToArg(depth_multiplier));

    // original filter_layout for depthwise is HWOI
    std::string cmsisnn_api = is_int16 ? "arm_convolve_wrapper_s16" : "arm_convolve_wrapper_s8";
    bool is_depthwise = depth_multiplier != -1;
    if (is_depthwise) {
      cmsisnn_api = is_int16 ? "arm_depthwise_conv_wrapper_s16" : "arm_depthwise_conv_wrapper_s8";
      int filter_pos_h = kernel_layout.find("H");
      int filter_pos_w = kernel_layout.find("W");
      Array<PrimExpr> depthwise_filter_shape{1, filter_shape[filter_pos_h],
                                             filter_shape[filter_pos_w], out_channels};
      filter_shape = depthwise_filter_shape;
    }
    int32_t filter_h = qnn::get_const_int(filter_shape[1]);
    int32_t filter_w = qnn::get_const_int(filter_shape[2]);

    tvm::Array<PrimExpr> call_ext_args = {tir::StringImm(cmsisnn_api), input, filter, multiplier};
    if (bias_add_call) {
      tir::Var bias = buffer_creator.GetBufferVar("bias");
      call_ext_args.push_back(bias);
    }
    call_ext_args.push_back(shift);
    call_ext_args.push_back(output);

    PrimExpr context_buffer_var = tir::StringImm("NULL");
    Target target = CreateTarget(transform::PassContext::Current());
    size_t context_buffer_size;
    if (is_depthwise) {
      context_buffer_size =
          DepthwiseConv2dBufferSize(is_int16, target, input_n, input_c, output_c, filter_w,
                                    filter_h, dilation_w, dilation_h, depth_multiplier);
    } else {
      context_buffer_size = Conv2dBufferSize(is_int16, target, padding_w, padding_h, input_n,
                                             input_h, input_c, output_h, output_w, stride_w,
                                             stride_h, dilation_w, dilation_h, filter_w, filter_h);
    }

    if (context_buffer_size) {
      String context_buffer_name = "context_buffer_" + std::to_string(context_buffer_id_++);
      context_buffer_var =
          tir::Var(context_buffer_name,
                   PointerType(PrimType(DataType::Int(context_buffer_bits)), "global.workspace"));
    }
    tvm::Array<PrimExpr> context_buffer_args = {context_buffer_var, ToArg(context_buffer_size)};

    scalar_args = tvm::runtime::Concat(context_buffer_args, scalar_args);
    scalar_args = tvm::runtime::Concat(scalar_args, input_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, filter_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, output_shape);
    call_ext_args = tvm::runtime::Concat(call_ext_args, scalar_args);

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), call_ext_args, context_buffer_var,
                            context_buffer_size, context_buffer_bits);
  }

  void EmitFullyConnected(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* clip_call = nullptr;
    const CallNode* requantize_call = nullptr;
    const CallNode* bias_add_call = nullptr;
    const CallNode* fc_call = nullptr;
    const CallNode* final_call = expr.as<CallNode>();
    const OpNode* final_op = final_call->op.as<OpNode>();
    if (final_op->name == "clip") {
      clip_call = final_call;
      requantize_call = clip_call->args[0].as<CallNode>();
    } else {
      requantize_call = final_call;
    }
    const CallNode* requantize_input = requantize_call->args[0].as<CallNode>();
    const OpNode* requantize_input_op = requantize_input->op.as<OpNode>();
    if (requantize_input_op->name == "nn.bias_add") {
      bias_add_call = requantize_input;
      fc_call = bias_add_call->args[0].as<CallNode>();
    } else {
      fc_call = requantize_input;
    }

    // Extract the size of the input parameter from the call arguments. Other params are based off
    // the input size
    int32_t dtype_bits = fc_call->args[0]->type_as<TensorTypeNode>()->dtype.bits();
    int32_t input_bits = dtype_bits;
    int32_t filter_bits = 8;
    int32_t bias_bits = dtype_bits * 4U;
    int32_t output_bits = dtype_bits;

    // TIR variables are created in the order they appear in the Relay partitioned function
    // %1 = qnn.dense(%input, %weight_const_0, input_zero_point_scalar, kernel_zero_point_scalar,
    //                 %input_scale_scalar, %kernel_scale_scalar)
    // %2 = nn.bias_add(%1, %bias_const_1, axis=1)
    // %3 = qnn.requantize(%2, %req_input_scale_scalar, %req_input_zero_point_scalar,
    //                     %output_scale_scalar, %output_zero_point_scalar)
    // clip(%3, a_min=%min_scalar, a_max=%max_scalar)
    BufferCreator buffer_creator;
    tir::Var input = buffer_creator.CreateBufferVar("input", DataType::Handle(input_bits));
    tir::Var filter = buffer_creator.CreateBufferVar("filter", DataType::Handle(filter_bits));
    if (bias_add_call) {
      buffer_creator.CreateBufferVar("bias", DataType::Handle(bias_bits));
    }
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(output_bits));

    // Individual arguments to the structs arguments of the CMSIS-NN API are filled into call_extern
    // https://github.com/ARM-software/CMSIS_5/blob/def6f800f95661eb3451d317f7d0dde504f6020d/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c#L50

    // prepare cmsis_nn_fc_params
    const DenseAttrs* dense_attrs = fc_call->attrs.as<DenseAttrs>();
    int32_t input_offset = -GetScalarFromConstant<int32_t>(fc_call->args[2]);
    int32_t filter_offset = -GetScalarFromConstant<int32_t>(fc_call->args[3]);
    int32_t output_offset = GetScalarFromConstant<int32_t>(requantize_call->args[4]);
    float input_scale = GetScalarFromConstant<float>(requantize_call->args[1]);
    float output_scale = GetScalarFromConstant<float>(requantize_call->args[3]);
    int32_t out_channels = qnn::get_const_int(dense_attrs->units);
    const auto [clip_min, clip_max] =
        clip_call ? GetClipMinMax(GetRef<Call>(clip_call)) : GetIntMinMax(dtype_bits);

    double quantized_multiplier =
        static_cast<double>(input_scale) / static_cast<double>(output_scale);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(quantized_multiplier);
    int32_t multiplier = std::get<0>(mult_shift_pair);
    int32_t shift = std::get<1>(mult_shift_pair);

    tvm::Array<PrimExpr> scalar_args = {
        ToArg(input_offset), ToArg(filter_offset), ToArg(output_offset), ToArg(clip_min),
        ToArg(clip_max),     ToArg(multiplier),    ToArg(shift)};

    Array<PrimExpr> input_shape = fc_call->args[0]->type_as<TensorTypeNode>()->shape;
    int32_t batch_size = qnn::get_const_int(input_shape[0]);
    int32_t in_channels = qnn::get_const_int(input_shape[1]);
    Array<PrimExpr> cmsisnn_input_shape{input_shape[0], 1, 1, input_shape[1]};

    Array<PrimExpr> cmsisnn_filter_shape{in_channels, 1, 1, out_channels};

    Array<PrimExpr> bias_shape{1, 1, 1, out_channels};

    Array<PrimExpr> cmsisnn_output_shape{batch_size, 1, 1, out_channels};

    std::string cmsisnn_api =
        dtype_bits == 16 ? "arm_fully_connected_s16" : "arm_fully_connected_s8";

    tvm::Array<PrimExpr> call_ext_args = {tir::StringImm(cmsisnn_api), input, filter};
    if (bias_add_call) {
      call_ext_args.push_back(buffer_creator.GetBufferVar("bias"));
    }
    call_ext_args.push_back(output);

    int context_buffer_size = 0;
    PrimExpr context_buffer_var = tir::StringImm("NULL");
    tvm::Array<PrimExpr> context_buffer_args = {context_buffer_var, ToArg(context_buffer_size)};

    scalar_args = tvm::runtime::Concat(context_buffer_args, scalar_args);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_input_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_filter_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_output_shape);
    call_ext_args = tvm::runtime::Concat(call_ext_args, scalar_args);

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), call_ext_args, context_buffer_var,
                            context_buffer_size);
  }

  void EmitPool2D(const GlobalVar& global_var, const Expr& expr, const String pool_name) {
    Call clip, pool;
    Call final_call = Downcast<Call>(expr);
    Op final_op = Downcast<Op>(final_call->op);
    if (final_op->name == "clip") {
      clip = final_call;
      Call clip_input = Downcast<Call>(clip->args[0]);
      Op clip_input_op = Downcast<Op>(clip_input->op);
      if (clip_input_op->name == "cast") {
        pool = Downcast<Call>(clip_input->args[0]);
      } else {  // max_pool2d
        pool = clip_input;
      }
    } else if (final_op->name == "cast") {
      pool = Downcast<Call>(final_call->args[0]);
    } else {  // max_pool2d
      pool = final_call;
    }

    int32_t dtype_bits = final_call->type_as<TensorTypeNode>()->dtype.bits();

    // prepare cmsis_nn_pool_params
    int32_t stride_h, stride_w, padding_h, padding_w, pool_size_h, pool_size_w;
    std::string cmsisnn_api;
    if (pool_name == "cmsis-nn.qnn_avg_pool2d") {
      if (dtype_bits == 8) {
        cmsisnn_api = "arm_avgpool_s8";
      } else {
        cmsisnn_api = "arm_avgpool_s16";
      }

      const AvgPool2DAttrs* attrs = pool->attrs.as<AvgPool2DAttrs>();
      stride_h = qnn::get_const_int(attrs->strides[0]);
      stride_w = qnn::get_const_int(attrs->strides[1]);
      padding_h = qnn::get_const_int(attrs->padding[0]);
      padding_w = qnn::get_const_int(attrs->padding[1]);
      pool_size_h = qnn::get_const_int(attrs->pool_size[0]);
      pool_size_w = qnn::get_const_int(attrs->pool_size[1]);
    } else {
      if (dtype_bits == 8) {
        cmsisnn_api = "arm_max_pool_s8";
      } else {
        cmsisnn_api = "arm_max_pool_s16";
      }

      const MaxPool2DAttrs* attrs = pool->attrs.as<MaxPool2DAttrs>();
      stride_h = qnn::get_const_int(attrs->strides[0]);
      stride_w = qnn::get_const_int(attrs->strides[1]);
      padding_h = qnn::get_const_int(attrs->padding[0]);
      padding_w = qnn::get_const_int(attrs->padding[1]);
      pool_size_h = qnn::get_const_int(attrs->pool_size[0]);
      pool_size_w = qnn::get_const_int(attrs->pool_size[1]);
    }

    const auto [clip_min, clip_max] =
        clip.defined() ? GetClipMinMax(clip) : GetIntMinMax(dtype_bits);

    tvm::Array<PrimExpr> scalar_args = {ToArg(stride_h),  ToArg(stride_w), ToArg(padding_h),
                                        ToArg(padding_w), ToArg(clip_min), ToArg(clip_max)};

    Array<PrimExpr> input_shape = pool->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> cmsisnn_input_shape{1, input_shape[1], input_shape[2], input_shape[3]};

    Array<PrimExpr> cmsisnn_filter_shape{1, pool_size_h, pool_size_w, 1};

    Array<PrimExpr> output_shape = pool->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> cmsisnn_output_shape{1, output_shape[1], output_shape[2], output_shape[3]};

    BufferCreator buffer_creator;
    tir::Var input = buffer_creator.CreateBufferVar("input", DataType::Handle(dtype_bits));
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(dtype_bits));
    tvm::Array<PrimExpr> call_ext_args = {tir::StringImm(cmsisnn_api), input, output};

    int context_buffer_size = 0;
    PrimExpr context_buffer_var = tir::StringImm("NULL");
    if (pool_name == "cmsis-nn.qnn_avg_pool2d") {
      Target target = CreateTarget(transform::PassContext::Current());
      int32_t input_c = qnn::get_const_int(input_shape[3]);
      context_buffer_size = AvgPoolBufferSize(target, input_c);
      if (context_buffer_size) {
        std::string context_buffer_name = "context_buffer_" + std::to_string(context_buffer_id_++);
        context_buffer_var = tir::Var(context_buffer_name,
                                      PointerType(PrimType(DataType::Int(8)), "global.workspace"));
      }
    }
    tvm::Array<PrimExpr> context_buffer_args = {context_buffer_var, ToArg(context_buffer_size)};

    scalar_args = tvm::runtime::Concat(context_buffer_args, scalar_args);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_input_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_filter_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_output_shape);
    call_ext_args = tvm::runtime::Concat(call_ext_args, scalar_args);

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), call_ext_args, context_buffer_var,
                            context_buffer_size);
  }

  void EmitSoftMax(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* quantize_call = expr.as<CallNode>();
    const CallNode* softmax_call = quantize_call->args[0].as<CallNode>();
    const CallNode* dequant_call = softmax_call->args[0].as<CallNode>();
    const float quant_scale = GetScalarFromConstant<float>(dequant_call->args[1]);

    // assuming layout as NHWC
    auto shape = quantize_call->type_as<TensorTypeNode>()->shape;
    int trailing_dim = shape.size() - 1;
    int row_size = shape[trailing_dim].as<tir::IntImmNode>()->value;
    int num_rows = 1;
    for (int i = 0; i < trailing_dim; ++i) {
      num_rows *= shape[i].as<tir::IntImmNode>()->value;
    }

    // calculate multiplier and shift for CMSIS-NN softmax API
    // Note: TensorFlow Lite Micro assumptions
    // Output zero point and scale are fixed to -128 and 1 / 256
    // kScaledDiffIntegerBits, kInputBits, kBeta are described on the following github page
    // https://github.com/tensorflow/tflite-micro/blob/d97cd0908d8cf5021e9d86f05a49888bee28c2a4/tensorflow/lite/micro/kernels/softmax_common.cc#L47
    double beta_multiplier = (kBeta * quant_scale * (1 << (31 - kInputBits)));
    beta_multiplier = std::min<double>(beta_multiplier, (1ll << 31) - 1.0);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(beta_multiplier);
    int32_t mult = std::get<0>(mult_shift_pair);
    int32_t shift = std::get<1>(mult_shift_pair);
    int32_t diff_min = (1 << kScaledDiffIntegerBits) - 1;
    diff_min <<= (31 - kScaledDiffIntegerBits);
    diff_min >>= shift;
    diff_min *= -1;

    BufferCreator buffer_creator;
    tir::Var in_var = buffer_creator.CreateBufferVar("input", DataType::Handle(8));
    tir::Var out_var = buffer_creator.CreateBufferVar("output", DataType::Handle(8));

    tvm::Array<PrimExpr> args = {
        tir::StringImm("arm_softmax_s8"),
        in_var,
        ToArg(num_rows),
        ToArg(row_size),
        ToArg(mult),
        ToArg(shift),
        ToArg(diff_min),
        out_var,
    };

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  struct BinaryElementwiseClipPattern {
    Call binary_op;
    Optional<Call> clip_op;
  };

  BinaryElementwiseClipPattern ParseBinaryElementwiseOpClipPattern(const Expr& expr) {
    BinaryElementwiseClipPattern pattern;
    Call final_call = Downcast<Call>(expr);
    const OpNode* final_op = final_call->op.as<OpNode>();
    if (final_op->name == "clip") {
      pattern.clip_op = final_call;
      pattern.binary_op = Downcast<Call>(final_call->args[0]);
    } else {
      pattern.binary_op = final_call;
      pattern.clip_op = Optional<Call>{nullptr};
    }
    return pattern;
  }

  void EmitMul(const GlobalVar& global_var, const Expr& expr) {
    const auto& pattern = ParseBinaryElementwiseOpClipPattern(expr);
    Call mul_call = pattern.binary_op;
    if (expr.as<CallNode>()->op.as<OpNode>()->name == "qnn.requantize")
      mul_call = GetRef<Call>(expr.as<CallNode>()->args[0].as<CallNode>());
    const auto bit_width = mul_call->type_as<TensorTypeNode>()->dtype.bits();
    const auto [output_min, output_max] =
        pattern.clip_op ? GetClipMinMax(pattern.clip_op.value()) : GetIntMinMax(bit_width);

    const float input_0_scale = GetScalarFromConstant<float>(mul_call->args[2]);
    const int32_t input_0_zero_point = GetScalarFromConstant<int32_t>(mul_call->args[3]);
    const float input_1_scale = GetScalarFromConstant<float>(mul_call->args[4]);
    const int32_t input_1_zero_point = GetScalarFromConstant<int32_t>(mul_call->args[5]);
    float output_scale = GetScalarFromConstant<float>(mul_call->args[6]);
    int32_t output_zero_point = GetScalarFromConstant<int32_t>(mul_call->args[7]);

    if (expr.as<CallNode>()->op.as<OpNode>()->name == "qnn.requantize") {
      output_scale = GetScalarFromConstant<float>(expr.as<CallNode>()->args[3]);
      output_zero_point = GetScalarFromConstant<int32_t>(expr.as<CallNode>()->args[4]);
    }

    double quantized_multiplier = static_cast<double>(input_0_scale) *
                                  static_cast<double>(input_1_scale) /
                                  static_cast<double>(output_scale);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(quantized_multiplier);
    int32_t output_multiplier = std::get<0>(mult_shift_pair);
    int32_t output_shift = std::get<1>(mult_shift_pair);

    PrimExpr tensor_size = mul_call->type_as<TensorTypeNode>()->Size();

    BufferCreator buffer_creator;
    tir::Var input_0 = buffer_creator.CreateBufferVar("input_0", DataType::Handle(bit_width));
    tir::Var input_1;
    if (mul_call->args[0].same_as(mul_call->args[1])) {
      input_1 = input_0;
    } else {
      input_1 = buffer_creator.CreateBufferVar("input_1", DataType::Handle(bit_width));
    }
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(bit_width));

    tvm::Array<PrimExpr> args = {
        tir::StringImm("arm_elementwise_mul_s" + std::to_string(bit_width)),
        input_0,
        input_1,
        ToArg(-input_0_zero_point),
        ToArg(-input_1_zero_point),
        output,
        ToArg(output_zero_point),
        ToArg(output_multiplier),
        ToArg(output_shift),
        ToArg(output_min),
        ToArg(output_max),
        tensor_size,
    };

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  void EmitAdd(const GlobalVar& global_var, const Expr& expr) {
    const auto& pattern = ParseBinaryElementwiseOpClipPattern(expr);
    Call add_call = pattern.binary_op;
    if (expr.as<CallNode>()->op.as<OpNode>()->name == "qnn.requantize")
      add_call = GetRef<Call>(expr.as<CallNode>()->args[0].as<CallNode>());
    const auto bit_width = add_call->type_as<TensorTypeNode>()->dtype.bits();

    const auto [output_min, output_max] =
        pattern.clip_op ? GetClipMinMax(pattern.clip_op.value()) : GetIntMinMax(bit_width);

    const float input_0_scale = GetScalarFromConstant<float>(add_call->args[2]);
    const int32_t input_0_zero_point = GetScalarFromConstant<int32_t>(add_call->args[3]);
    const float input_1_scale = GetScalarFromConstant<float>(add_call->args[4]);
    const int32_t input_1_zero_point = GetScalarFromConstant<int32_t>(add_call->args[5]);
    float output_scale = GetScalarFromConstant<float>(add_call->args[6]);
    int32_t output_zero_point = GetScalarFromConstant<int32_t>(add_call->args[7]);

    const int32_t left_shift = (bit_width == 16) ? 15 : 20;
    const int32_t input_0_offset = -input_0_zero_point;
    const int32_t input_1_offset = -input_1_zero_point;
    const int32_t output_offset = output_zero_point;

    if (expr.as<CallNode>()->op.as<OpNode>()->name == "qnn.requantize") {
      output_scale = GetScalarFromConstant<float>(expr.as<CallNode>()->args[3]);
      output_zero_point = GetScalarFromConstant<int32_t>(expr.as<CallNode>()->args[4]);
    }

    const float max_input_scale = std::max(input_0_scale, input_1_scale);
    const double twice_max_input_scale = 2 * static_cast<double>(max_input_scale);
    const double scaled_input_0_scale = static_cast<double>(input_0_scale) / twice_max_input_scale;
    const double scaled_input_1_scale = static_cast<double>(input_1_scale) / twice_max_input_scale;
    const double scaled_output_scale =
        twice_max_input_scale / ((1 << left_shift) * static_cast<double>(output_scale));

    auto input_0_mult_shift_pair =
        tvm::relay::qnn::GetFixedPointMultiplierShift(scaled_input_0_scale);
    int32_t input_0_multiplier = std::get<0>(input_0_mult_shift_pair);
    int32_t input_0_shift = std::get<1>(input_0_mult_shift_pair);

    auto input_1_mult_shift_pair =
        tvm::relay::qnn::GetFixedPointMultiplierShift(scaled_input_1_scale);
    int32_t input_1_multiplier = std::get<0>(input_1_mult_shift_pair);
    int32_t input_1_shift = std::get<1>(input_1_mult_shift_pair);

    auto output_mult_shift_pair =
        tvm::relay::qnn::GetFixedPointMultiplierShift(scaled_output_scale);
    int32_t output_multiplier = std::get<0>(output_mult_shift_pair);
    int32_t output_shift = std::get<1>(output_mult_shift_pair);

    PrimExpr tensor_size = add_call->type_as<TensorTypeNode>()->Size();

    BufferCreator buffer_creator;
    tir::Var input_0 = buffer_creator.CreateBufferVar("input_0", DataType::Handle(bit_width));
    tir::Var input_1;
    if (add_call->args[0].same_as(add_call->args[1])) {
      input_1 = input_0;
    } else {
      input_1 = buffer_creator.CreateBufferVar("input_1", DataType::Handle(bit_width));
    }
    tir::Var output = buffer_creator.CreateBufferVar("output", DataType::Handle(bit_width));

    tvm::Array<PrimExpr> args = {
        tir::StringImm("arm_elementwise_add_s" + std::to_string(bit_width)),
        input_0,
        input_1,
        ToArg(input_0_offset),
        ToArg(input_0_multiplier),
        ToArg(input_0_shift),
        ToArg(input_1_offset),
        ToArg(input_1_multiplier),
        ToArg(input_1_shift),
        ToArg(left_shift),
        output,
        ToArg(output_offset),
        ToArg(output_multiplier),
        ToArg(output_shift),
        ToArg(output_min),
        ToArg(output_max),
        tensor_size,
    };

    CreatePrimFuncForExtern(global_var, buffer_creator.GetPrimFuncParams(),
                            buffer_creator.GetBufferMap(), args);
  }

  // Removes kCompiler attribute from the partitioned functions that are not supported by this
  // RelayToTIR
  Call CallToFuncWithoutCompilerAttr(GlobalVar new_global_var, Call call, Function func) {
    Function new_func = WithoutAttr(std::move(func), attr::kCompiler);
    ir_module_->Update(new_global_var, new_func);
    return Call(new_global_var, call->args, call->attrs, call->type_args, call->span);
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      Expr var = this->VisitExpr(op->var);
      Expr value = this->VisitExpr(op->value);
      // outlineable function no longer needs let binding
      if (this->CanOutlineExpr(value)) {
        this->memo_[var] = value;
      }
    };
    auto post_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);
      // drop the let binding
      if (this->CanOutlineExpr(value)) {
        this->memo_[expr] = this->VisitExpr(op->body);
      } else {
        Var var = Downcast<Var>(this->VisitExpr(op->var));
        if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(var, value, body);
        }
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  bool CanOutlineExpr(const Expr& expr) {
    // TODO(@lhutton1): This behaviour is similar to the OutlineCompilerFunctions pass
    // we could reuse this functionality by separating outlining and lowering in this
    // pass.
    if (!expr->IsInstance<FunctionNode>()) {
      return false;
    }
    const auto* func = expr.as<FunctionNode>();
    auto codegen_name = func->GetAttr<String>(attr::kCompiler);
    if (!codegen_name.defined() || codegen_name != "cmsis-nn") {
      return false;
    }
    return true;
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const auto* call = post.as<CallNode>()) {
      if (CanOutlineExpr(call->op)) {
        const auto* func = call->op.as<FunctionNode>();
        ICHECK(func) << "Expected function node but was " << call->op->GetTypeKey();
        const auto codegen_name = func->GetAttr<String>(attr::kCompiler);
        auto global_func_name = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        GlobalVar new_global_var(global_func_name.value());

        const CallNode* inner_call = func->body.as<CallNode>();
        if (!inner_call) {
          return CallToFuncWithoutCompilerAttr(new_global_var, GetRef<Call>(call),
                                               GetRef<Function>(func));
        }

        const FunctionNode* composite_func = inner_call->op.as<FunctionNode>();
        if (!composite_func) {
          return CallToFuncWithoutCompilerAttr(new_global_var, GetRef<Call>(call),
                                               GetRef<Function>(func));
        }

        auto comp_name = composite_func->GetAttr<String>(attr::kComposite);
        new_global_var->checked_type_ = composite_func->checked_type();

        if (comp_name == "cmsis-nn.qnn_softmax") {
          EmitSoftMax(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.qnn_mul" or comp_name == "cmsis-nn.qnn_mul2") {
          EmitMul(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.qnn_add") {
          EmitAdd(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.qnn_conv2d") {
          EmitConv2D(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.qnn_fully_connected") {
          EmitFullyConnected(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.qnn_avg_pool2d" ||
                   comp_name == "cmsis-nn.qnn_max_pool2d") {
          EmitPool2D(new_global_var, composite_func->body, comp_name.value());
        } else if (comp_name == "cmsis-nn.fiti_sigmoid") {
          Fiti_Sigmoid(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.fiti_reduce_transaction1") {
          fiti_reduce_transaction1(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.fiti_reduce_transaction2") {
          fiti_reduce_transaction2(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.fiti_resize2d") {
          Fiti_Resize2d(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.fiti_requant") {
          Fiti_Requant(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.fiti_concat") {
          Fiti_Concat(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.fiti_mamm") {
          Fiti_MAMM(new_global_var, composite_func->body);
        } else if (comp_name == "cmsis-nn.fiti_slice") {
          Fiti_Slice(new_global_var, composite_func->body);
        } else {
          return CallToFuncWithoutCompilerAttr(new_global_var, GetRef<Call>(call),
                                               GetRef<Function>(func));
        }

        // Drop out the redundant arguments, and the arg_types from the global function call
        Array<Expr> args;
        Array<Type> arg_types;
        auto* func_type = new_global_var->checked_type_.as<FuncTypeNode>();
        int arg_id = -1;
        for (const auto& arg : call->args) {
          ++arg_id;
          if (std::find(skip_call_args_.begin(), skip_call_args_.end(), arg_id) !=
              skip_call_args_.end()) {
            continue;
          }
          args.push_back(VisitExpr(arg));
          arg_types.push_back(func_type->arg_types[arg_id]);
        }
        if (arg_types.size() != func_type->arg_types.size()) {
          new_global_var->checked_type_ =
              FuncType(arg_types, func_type->ret_type, {}, func_type->type_constraints);
        }
        skip_call_args_.clear();
        return Call(new_global_var, args, call->attrs, call->type_args, call->span);
      }
    }
    return post;
  }

 private:
  static constexpr int32_t kScaledDiffIntegerBits = 5;
  static constexpr int32_t kInputBits = 5;
  static constexpr double kBeta = 1.0;
  /*! \brief Unique id for context buffer needed by CMSIS-NN layers. */
  int32_t context_buffer_id_;
  /*! \brief Skip arguments in the call to global partitioned function. */
  std::unordered_set<int32_t> skip_call_args_;
  IRModule ir_module_;
  Target target_;
};

tvm::transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule ir_module, transform::PassContext pass_context) {
        auto relay_to_tir = RelayToTIRVisitor(ir_module, Target("cmsis-nn"));
        return relay_to_tir.Mutate();
      };
  return tvm::transform::CreateModulePass(pass_func, 0, "RelayToTIR", {});
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
