
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
/*!
 * \file generate_constant.cc
 * \brief Generates quantization parameters needed by CMSIS-NN
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>

#include "../../../op/make_op.h"
#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"
#include "../constant_transforms.h"
#include "convolutions.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*!
 * \brief This Mutator will find all partitioned functions meant for CMSIS-NN Conv2D.
 * It will substitute original Conv2D's weight zero point and original Requantize's input zero point
 * with CMSIS-NN's quantization parameters.
 * https://github.com/tensorflow/tflite-micro/blob/0f40100fc60276e9f345c23282de3baf19a78059/tensorflow/lite/kernels/internal/quantization_util.cc#L53
 */
class GenerateConstantsMutator : public MixedModeMutator {
 public:
  explicit GenerateConstantsMutator(const IRModule& mod) : mod_(mod) {}

 private:
  /*!  * \brief Converts Kernel layout from HWIO to OHWI to align to CMSIS-NN requirements */
  Expr ConvertKernelLayout(Expr kernel_expr, const Conv2DAttrs* conv2d_attrs, Attrs* new_attrs) {
    auto attrs = make_object<Conv2DAttrs>();
    attrs->strides = std::move(conv2d_attrs->strides);
    attrs->padding = std::move(conv2d_attrs->padding);
    attrs->dilation = std::move(conv2d_attrs->dilation);
    attrs->groups = conv2d_attrs->groups;
    attrs->channels = std::move(conv2d_attrs->channels);
    attrs->kernel_size = std::move(conv2d_attrs->kernel_size);
    attrs->data_layout = std::move(conv2d_attrs->data_layout);
    attrs->kernel_layout = runtime::String("OHWI");
    attrs->out_layout = std::move(conv2d_attrs->out_layout);
    attrs->out_dtype = std::move(conv2d_attrs->out_dtype);
    *new_attrs = tvm::Attrs{attrs};

    Constant conv2d_kernel = Downcast<Constant>(kernel_expr);
    conv2d_kernel = TransposeWeights(conv2d_kernel, conv2d_attrs->kernel_layout, "OHWI");
    return conv2d_kernel;
  }

  /*!  * \brief Performs weight transpose and substitutes existing constants in the composite
   *            function for Conv2D with CMSIS-NN Requantize constants */
  Expr GenerateConv2dRequantConstants(const Expr& expr) {
    const CallNode* clip_call = nullptr;
    const CallNode* requantize_call = nullptr;
    const CallNode* bias_add_call = nullptr;
    const CallNode* conv2d_call = nullptr;
    auto* final_call = expr.as<CallNode>();
    auto* final_op = final_call->op.as<OpNode>();
    if (final_op->name == "qnn.requantize") {
      if (final_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "clip") {
        clip_call = final_call->args[0].as<CallNode>();
        requantize_call = clip_call->args[0].as<CallNode>();
      } else
        requantize_call = final_call;
    } else if (final_op->name == "clip") {
      clip_call = final_call;
      requantize_call = clip_call->args[0].as<CallNode>();
    } else {
      requantize_call = final_call;
    }
    auto* requantize_input = requantize_call->args[0].as<CallNode>();
    auto* requantize_input_op = requantize_input->op.as<OpNode>();
    if (requantize_input_op->name == "nn.bias_add") {
      bias_add_call = requantize_input;
      conv2d_call = bias_add_call->args[0].as<CallNode>();
    } else {
      conv2d_call = requantize_input;
    }

    auto* conv2d_attrs = conv2d_call->attrs.as<Conv2DAttrs>();
    tvm::Attrs new_conv2d_attrs = conv2d_call->attrs;
    Expr conv2d_kernel = conv2d_call->args[1];

    Array<PrimExpr> input_shape = conv2d_call->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> kernel_shape = conv2d_call->args[1]->type_as<TensorTypeNode>()->shape;
    if (!IsCMSISNNDepthwise(conv2d_attrs, input_shape, kernel_shape)) {
      // Transpose weights: HWIO -> OHWI for Conv2D
      conv2d_kernel = ConvertKernelLayout(conv2d_call->args[1], conv2d_attrs, &new_conv2d_attrs);
    }

    // Obtain input and output scales from Relay's Requantization
    int64_t out_channels = conv2d_attrs->channels.as<IntImmNode>()->value;
    float output_scale = GetScalarFromConstant<float>(requantize_call->args[3]);
    if (final_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "clip")
      output_scale = GetScalarFromConstant<float>(final_call->args[3]);
    auto input_scale = GetScalarFromConstant<float>(conv2d_call->args[4]);
    auto filter_scales = tvm::relay::qnn::GetFloatVectorFromConstant(conv2d_call->args[5]);

    // Calculate requantization multiplier and shift
    Device dev{DLDeviceType::kDLCPU, 0};
    runtime::NDArray multiplier_nda =
        runtime::NDArray::Empty({out_channels}, DataType::Int(32), dev);
    runtime::NDArray shift_nda = runtime::NDArray::Empty({out_channels}, DataType::Int(32), dev);
    int32_t* multiplier = static_cast<int32_t*>(multiplier_nda->data);
    int32_t* shift = static_cast<int32_t*>(shift_nda->data);
    for (int i = 0; i < out_channels; ++i) {
      double effective_output_scale =
          static_cast<double>(input_scale) * filter_scales[i] / static_cast<double>(output_scale);
      std::tie(*(multiplier + i), *(shift + i)) =
          tvm::relay::qnn::GetFixedPointMultiplierShift(effective_output_scale);
    }

    // Create constants from requantization multiplier and shift
    Constant multiplier_const(multiplier_nda);
    Constant shift_const(shift_nda);

    // Convert scale scalars into Constants
    // Scales are expected as Constants by following passes
    Expr weight_scale = conv2d_call->args[5];
    Expr req_inp_scale = requantize_call->args[1];
    if (out_channels == 1) {
      runtime::NDArray weight_scale_nda =
          runtime::NDArray::Empty({out_channels}, DataType::Float(32), dev);
      float* weight_scale_p = static_cast<float*>(weight_scale_nda->data);
      *weight_scale_p = GetScalarFromConstant<float>(weight_scale);
      weight_scale = Constant(weight_scale_nda);

      runtime::NDArray req_inp_scale_nda =
          runtime::NDArray::Empty({out_channels}, DataType::Float(32), dev);
      float* req_inp_scale_p = static_cast<float*>(req_inp_scale_nda->data);
      *req_inp_scale_p = GetScalarFromConstant<float>(req_inp_scale);
      req_inp_scale = Constant(req_inp_scale_nda);
    }

    // Replace existing weights (HWIO) with the transposed ones (OHWI) for Conv2D
    // Substitute Conv2D weight_zero_point with the CMSIS-NN multiplier
    // Substitute Requantize input_zero_point with CMSIS-NN shift
    // Conv2D arguments: data, weight, input_zp, weight_zp, input_sc, weight_sc
    Array<Expr> conv2d_args = {conv2d_call->args[0], conv2d_kernel,        conv2d_call->args[2],
                               multiplier_const,     conv2d_call->args[4], weight_scale};
    Call ret_call = Call(conv2d_call->op, conv2d_args, new_conv2d_attrs, {}, conv2d_call->span);
    if (bias_add_call) {
      ret_call = Call(bias_add_call->op, {ret_call, bias_add_call->args[1]}, bias_add_call->attrs,
                      {}, bias_add_call->span);
    }
    Array<Expr> requantize_args = {ret_call, req_inp_scale, shift_const, requantize_call->args[3],
                                   requantize_call->args[4]};
    if (final_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "clip")
      requantize_args = {ret_call, req_inp_scale, shift_const, final_call->args[3],
                         final_call->args[4]};
    ret_call = Call(requantize_call->op, requantize_args, requantize_call->attrs, {},
                    requantize_call->span);
    if (clip_call) {
      ret_call = Call(clip_call->op, {ret_call}, clip_call->attrs, {}, clip_call->span);
    }
    return std::move(ret_call);
  }

  Expr GenerateConv2dRequantConstants_1(const Expr& expr) {
    // conv - conv - conv
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
    auto* final_call = expr.as<CallNode>();
    auto* final_op = final_call->op.as<OpNode>();

    if (final_op->name == "qnn.requantize") {
      if (final_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "clip") {
        clip_call3 = final_call->args[0].as<CallNode>();
        requantize_call3 = clip_call3->args[0].as<CallNode>();
      } else
        requantize_call3 = final_call;
    } else if (final_op->name == "clip") {
      clip_call3 = final_call;
      requantize_call3 = clip_call3->args[0].as<CallNode>();
    } else
      requantize_call3 = final_call;

    auto* requantize3_input = requantize_call3->args[0].as<CallNode>();
    auto* requantize3_input_op = requantize3_input->op.as<OpNode>();
    if (requantize3_input_op->name == "nn.bias_add") {
      bias_add_call3 = requantize3_input;
      conv2d_call3 = bias_add_call3->args[0].as<CallNode>();
    } else
      conv2d_call3 = requantize3_input;

    if ((conv2d_call3->args[0]).as<CallNode>()->op.as<OpNode>()->name == "clip") {
      clip_call2 = conv2d_call3->args[0].as<CallNode>();
      requantize_call2 = clip_call2->args[0].as<CallNode>();
    } else
      requantize_call2 = conv2d_call3->args[0].as<CallNode>();

    auto* requantize2_input = requantize_call2->args[0].as<CallNode>();
    auto* requantize2_input_op = requantize2_input->op.as<OpNode>();
    if (requantize2_input_op->name == "nn.bias_add") {
      bias_add_call2 = requantize2_input;
      conv2d_call2 = bias_add_call2->args[0].as<CallNode>();
    } else
      conv2d_call2 = requantize2_input;

    if ((conv2d_call2->args[0]).as<CallNode>()->op.as<OpNode>()->name == "clip") {
      clip_call1 = (conv2d_call2->args[0]).as<CallNode>();
      requantize_call1 = (clip_call1->args[0]).as<CallNode>();
    } else
      requantize_call1 = conv2d_call2->args[0].as<CallNode>();

    auto* requantize1_input = (requantize_call1->args[0]).as<CallNode>();
    auto* requantize1_input_op = requantize1_input->op.as<OpNode>();
    if (requantize1_input_op->name == "nn.bias_add") {
      bias_add_call1 = requantize1_input;
      conv2d_call1 = (bias_add_call1->args[0]).as<CallNode>();
    } else
      conv2d_call1 = requantize1_input;

    auto* conv2d_attrs1 = conv2d_call1->attrs.as<Conv2DAttrs>();
    tvm::Attrs new_conv2d_attrs1 = conv2d_call1->attrs;
    Expr conv2d_kernel1 = conv2d_call1->args[1];

    auto* conv2d_attrs2 = conv2d_call2->attrs.as<Conv2DAttrs>();
    tvm::Attrs new_conv2d_attrs2 = conv2d_call2->attrs;
    Expr conv2d_kernel2 = conv2d_call2->args[1];

    auto* conv2d_attrs3 = conv2d_call3->attrs.as<Conv2DAttrs>();
    tvm::Attrs new_conv2d_attrs3 = conv2d_call3->attrs;
    Expr conv2d_kernel3 = conv2d_call3->args[1];

    Array<PrimExpr> input_shape1 = conv2d_call1->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> kernel_shape1 = conv2d_call1->args[1]->type_as<TensorTypeNode>()->shape;
    if (!IsCMSISNNDepthwise(conv2d_attrs1, input_shape1, kernel_shape1))
      conv2d_kernel1 =
          ConvertKernelLayout(conv2d_call1->args[1], conv2d_attrs1, &new_conv2d_attrs1);

    Array<PrimExpr> input_shape2 = conv2d_call2->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> kernel_shape2 = conv2d_call2->args[1]->type_as<TensorTypeNode>()->shape;
    if (!IsCMSISNNDepthwise(conv2d_attrs2, input_shape2, kernel_shape2))
      conv2d_kernel2 =
          ConvertKernelLayout(conv2d_call2->args[1], conv2d_attrs2, &new_conv2d_attrs2);

    Array<PrimExpr> input_shape3 = conv2d_call3->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> kernel_shape3 = conv2d_call3->args[1]->type_as<TensorTypeNode>()->shape;
    if (!IsCMSISNNDepthwise(conv2d_attrs3, input_shape3, kernel_shape3))
      conv2d_kernel3 =
          ConvertKernelLayout(conv2d_call3->args[1], conv2d_attrs3, &new_conv2d_attrs3);

    // Obtain input and output scales from Relay's Requantization
    int64_t out_channels1 = conv2d_attrs1->channels.as<IntImmNode>()->value;
    float output_scale1 = GetScalarFromConstant<float>(requantize_call1->args[3]);
    auto input_scale1 = GetScalarFromConstant<float>(conv2d_call1->args[4]);
    auto filter_scales1 = tvm::relay::qnn::GetFloatVectorFromConstant(conv2d_call1->args[5]);

    // Calculate requantization multiplier and shift
    Device dev{DLDeviceType::kDLCPU, 0};
    runtime::NDArray multiplier1_nda =
        runtime::NDArray::Empty({out_channels1}, DataType::Int(32), dev);
    runtime::NDArray shift1_nda = runtime::NDArray::Empty({out_channels1}, DataType::Int(32), dev);
    int32_t* multiplier1 = static_cast<int32_t*>(multiplier1_nda->data);
    int32_t* shift1 = static_cast<int32_t*>(shift1_nda->data);
    for (int i = 0; i < out_channels1; ++i) {
      double effective_output_scale1 = static_cast<double>(input_scale1) * filter_scales1[i] /
                                       static_cast<double>(output_scale1);
      std::tie(*(multiplier1 + i), *(shift1 + i)) =
          tvm::relay::qnn::GetFixedPointMultiplierShift(effective_output_scale1);
    }

    int64_t out_channels2 = conv2d_attrs2->channels.as<IntImmNode>()->value;
    float output_scale2 = GetScalarFromConstant<float>(requantize_call2->args[3]);
    auto input_scale2 = GetScalarFromConstant<float>(conv2d_call2->args[4]);
    auto filter_scales2 = tvm::relay::qnn::GetFloatVectorFromConstant(conv2d_call2->args[5]);

    runtime::NDArray multiplier2_nda =
        runtime::NDArray::Empty({out_channels2}, DataType::Int(32), dev);
    runtime::NDArray shift2_nda = runtime::NDArray::Empty({out_channels2}, DataType::Int(32), dev);
    int32_t* multiplier2 = static_cast<int32_t*>(multiplier2_nda->data);
    int32_t* shift2 = static_cast<int32_t*>(shift2_nda->data);
    for (int i = 0; i < out_channels2; ++i) {
      double effective_output_scale2 = static_cast<double>(input_scale2) * filter_scales2[i] /
                                       static_cast<double>(output_scale2);
      std::tie(*(multiplier2 + i), *(shift2 + i)) =
          tvm::relay::qnn::GetFixedPointMultiplierShift(effective_output_scale2);
    }

    int64_t out_channels3 = conv2d_attrs3->channels.as<IntImmNode>()->value;
    float output_scale3 = GetScalarFromConstant<float>(requantize_call3->args[3]);
    auto input_scale3 = GetScalarFromConstant<float>(conv2d_call3->args[4]);
    auto filter_scales3 = tvm::relay::qnn::GetFloatVectorFromConstant(conv2d_call3->args[5]);

    runtime::NDArray multiplier3_nda =
        runtime::NDArray::Empty({out_channels3}, DataType::Int(32), dev);
    runtime::NDArray shift3_nda = runtime::NDArray::Empty({out_channels3}, DataType::Int(32), dev);
    int32_t* multiplier3 = static_cast<int32_t*>(multiplier3_nda->data);
    int32_t* shift3 = static_cast<int32_t*>(shift3_nda->data);
    for (int i = 0; i < out_channels3; ++i) {
      double effective_output_scale3 = static_cast<double>(input_scale3) * filter_scales3[i] /
                                       static_cast<double>(output_scale3);
      std::tie(*(multiplier3 + i), *(shift3 + i)) =
          tvm::relay::qnn::GetFixedPointMultiplierShift(effective_output_scale3);
    }

    // Create constants from requantization multiplier and shift
    Constant multiplier1_const(multiplier1_nda);
    Constant shift1_const(shift1_nda);
    Constant multiplier2_const(multiplier2_nda);
    Constant shift2_const(shift2_nda);
    Constant multiplier3_const(multiplier3_nda);
    Constant shift3_const(shift3_nda);

    // Convert scale scalars into Constants
    // Scales are expected as Constants by following passes
    Expr weight1_scale = conv2d_call1->args[5];
    Expr req1_inp_scale = requantize_call1->args[1];
    if (out_channels1 == 1) {
      runtime::NDArray weight1_scale_nda =
          runtime::NDArray::Empty({out_channels1}, DataType::Float(32), dev);
      float* weight1_scale_p = static_cast<float*>(weight1_scale_nda->data);
      *weight1_scale_p = GetScalarFromConstant<float>(weight1_scale);
      weight1_scale = Constant(weight1_scale_nda);

      runtime::NDArray req1_inp_scale_nda =
          runtime::NDArray::Empty({out_channels1}, DataType::Float(32), dev);
      float* req1_inp_scale_p = static_cast<float*>(req1_inp_scale_nda->data);
      *req1_inp_scale_p = GetScalarFromConstant<float>(req1_inp_scale);
      req1_inp_scale = Constant(req1_inp_scale_nda);
    }

    Expr weight2_scale = conv2d_call2->args[5];
    Expr req2_inp_scale = requantize_call2->args[1];
    if (out_channels2 == 1) {
      runtime::NDArray weight2_scale_nda =
          runtime::NDArray::Empty({out_channels2}, DataType::Float(32), dev);
      float* weight2_scale_p = static_cast<float*>(weight2_scale_nda->data);
      *weight2_scale_p = GetScalarFromConstant<float>(weight2_scale);
      weight2_scale = Constant(weight2_scale_nda);

      runtime::NDArray req2_inp_scale_nda =
          runtime::NDArray::Empty({out_channels2}, DataType::Float(32), dev);
      float* req2_inp_scale_p = static_cast<float*>(req2_inp_scale_nda->data);
      *req2_inp_scale_p = GetScalarFromConstant<float>(req2_inp_scale);
      req2_inp_scale = Constant(req2_inp_scale_nda);
    }

    Expr weight3_scale = conv2d_call3->args[5];
    Expr req3_inp_scale = requantize_call3->args[1];
    if (out_channels3 == 1) {
      runtime::NDArray weight3_scale_nda =
          runtime::NDArray::Empty({out_channels3}, DataType::Float(32), dev);
      float* weight3_scale_p = static_cast<float*>(weight3_scale_nda->data);
      *weight3_scale_p = GetScalarFromConstant<float>(weight3_scale);
      weight3_scale = Constant(weight3_scale_nda);

      runtime::NDArray req3_inp_scale_nda =
          runtime::NDArray::Empty({out_channels3}, DataType::Float(32), dev);
      float* req3_inp_scale_p = static_cast<float*>(req3_inp_scale_nda->data);
      *req3_inp_scale_p = GetScalarFromConstant<float>(req3_inp_scale);
      req3_inp_scale = Constant(req3_inp_scale_nda);
    }
    // Replace existing weights (HWIO) with the transposed ones (OHWI) for Conv2D
    // Substitute Conv2D weight_zero_point with the CMSIS-NN multiplier
    // Substitute Requantize input_zero_point with CMSIS-NN shift
    // Conv2D arguments: data, weight, input_zp, weight_zp, input_sc, weight_sc
    Array<Expr> conv2d_args1 = {conv2d_call1->args[0], conv2d_kernel1,        conv2d_call1->args[2],
                                multiplier1_const,     conv2d_call1->args[4], weight1_scale};
    Call ret_call = Call(conv2d_call1->op, conv2d_args1, new_conv2d_attrs1, {});
    if (bias_add_call1)
      ret_call =
          Call(bias_add_call1->op, {ret_call, bias_add_call1->args[1]}, bias_add_call1->attrs, {});
    Array<Expr> requantize_args1 = {ret_call, req1_inp_scale, shift1_const,
                                    requantize_call1->args[3], requantize_call1->args[4]};
    ret_call = Call(requantize_call1->op, requantize_args1, requantize_call1->attrs, {});
    if (clip_call1) ret_call = Call(clip_call1->op, {ret_call}, clip_call1->attrs, {});
    Array<Expr> conv2d_args2 = {ret_call,          conv2d_kernel2,        conv2d_call2->args[2],
                                multiplier2_const, conv2d_call2->args[4], weight2_scale};
    ret_call = Call(conv2d_call2->op, conv2d_args2, new_conv2d_attrs2, {});
    if (bias_add_call2)
      ret_call =
          Call(bias_add_call2->op, {ret_call, bias_add_call2->args[1]}, bias_add_call2->attrs, {});
    Array<Expr> requantize_args2 = {ret_call, req2_inp_scale, shift2_const,
                                    requantize_call2->args[3], requantize_call2->args[4]};
    ret_call = Call(requantize_call2->op, requantize_args2, requantize_call2->attrs, {});
    if (clip_call2) ret_call = Call(clip_call2->op, {ret_call}, clip_call2->attrs, {});
    Array<Expr> conv2d_args3 = {ret_call,          conv2d_kernel3,        conv2d_call3->args[2],
                                multiplier3_const, conv2d_call3->args[4], weight3_scale};
    ret_call = Call(conv2d_call3->op, conv2d_args3, new_conv2d_attrs3, {});
    if (bias_add_call3)
      ret_call =
          Call(bias_add_call3->op, {ret_call, bias_add_call3->args[1]}, bias_add_call3->attrs, {});
    Array<Expr> requantize_args3 = {ret_call, req3_inp_scale, shift3_const,
                                    requantize_call3->args[3], requantize_call3->args[4]};
    ret_call = Call(requantize_call3->op, requantize_args3, requantize_call3->attrs, {});
    if (clip_call3) ret_call = Call(clip_call3->op, {ret_call}, clip_call3->attrs, {});

    return std::move(ret_call);
  }

  Expr GenerateConv2dRequantConstants_2(const Expr& expr) {
    //(pad) - conv - (pad) - conv
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
    auto* final_call = expr.as<CallNode>();
    auto* final_op = final_call->op.as<OpNode>();

    if (final_op->name == "qnn.requantize") {
      if (final_call->args[0].as<CallNode>()->op.as<OpNode>()->name == "clip") {
        clip_call2 = final_call->args[0].as<CallNode>();
        requantize_call2 = clip_call2->args[0].as<CallNode>();
      } else
        requantize_call2 = final_call;
    } else if (final_op->name == "clip") {
      clip_call2 = final_call;
      requantize_call2 = clip_call2->args[0].as<CallNode>();
    } else
      requantize_call2 = final_call;

    auto* requantize2_input = requantize_call2->args[0].as<CallNode>();
    auto* requantize2_input_op = requantize2_input->op.as<OpNode>();
    if (requantize2_input_op->name == "nn.bias_add") {
      bias_add_call2 = requantize2_input;
      conv2d_call2 = bias_add_call2->args[0].as<CallNode>();
    } else
      conv2d_call2 = requantize2_input;

    if ((conv2d_call2->args[0]).as<CallNode>()->op.as<OpNode>()->name == "clip") {
      clip_call1 = (conv2d_call2->args[0]).as<CallNode>();
      requantize_call1 = (clip_call1->args[0]).as<CallNode>();
    } else {
      pad_call2 = conv2d_call2->args[0].as<CallNode>();
      clip_call1 = pad_call2->args[0].as<CallNode>();
      requantize_call1 = clip_call1->args[0].as<CallNode>();
    }

    auto* requantize1_input = (requantize_call1->args[0]).as<CallNode>();
    auto* requantize1_input_op = requantize1_input->op.as<OpNode>();
    if (requantize1_input_op->name == "nn.bias_add") {
      bias_add_call1 = requantize1_input;
      conv2d_call1 = (bias_add_call1->args[0]).as<CallNode>();
    } else
      conv2d_call1 = requantize1_input;

    if (conv2d_call1->args[0]->IsInstance<CallNode>() &&
        (conv2d_call1->args[0]).as<CallNode>()->op.as<OpNode>()->name == "nn.pad")
      pad_call1 = conv2d_call1->args[0].as<CallNode>();

    auto* conv2d_attrs1 = conv2d_call1->attrs.as<Conv2DAttrs>();
    tvm::Attrs new_conv2d_attrs1 = conv2d_call1->attrs;
    Expr conv2d_kernel1 = conv2d_call1->args[1];

    auto* conv2d_attrs2 = conv2d_call2->attrs.as<Conv2DAttrs>();
    tvm::Attrs new_conv2d_attrs2 = conv2d_call2->attrs;
    Expr conv2d_kernel2 = conv2d_call2->args[1];

    Array<PrimExpr> input_shape1 = conv2d_call1->args[0]->type_as<TensorTypeNode>()->shape;
    if (pad_call1) input_shape1 = pad_call1->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> kernel_shape1 = conv2d_call1->args[1]->type_as<TensorTypeNode>()->shape;
    if (!IsCMSISNNDepthwise(conv2d_attrs1, input_shape1, kernel_shape1))
      conv2d_kernel1 =
          ConvertKernelLayout(conv2d_call1->args[1], conv2d_attrs1, &new_conv2d_attrs1);

    Array<PrimExpr> input_shape2 = conv2d_call2->args[0]->type_as<TensorTypeNode>()->shape;
    if (pad_call2) input_shape2 = pad_call2->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> kernel_shape2 = conv2d_call2->args[1]->type_as<TensorTypeNode>()->shape;
    if (!IsCMSISNNDepthwise(conv2d_attrs2, input_shape2, kernel_shape2))
      conv2d_kernel2 =
          ConvertKernelLayout(conv2d_call2->args[1], conv2d_attrs2, &new_conv2d_attrs2);

    // Obtain input and output scales from Relay's Requantization
    int64_t out_channels1 = conv2d_attrs1->channels.as<IntImmNode>()->value;
    float output_scale1 = GetScalarFromConstant<float>(requantize_call1->args[3]);
    // output_scale1 = GetScalarFromConstant<float>(clip_call1->args[3]);

    auto input_scale1 = GetScalarFromConstant<float>(conv2d_call1->args[4]);
    auto filter_scales1 = tvm::relay::qnn::GetFloatVectorFromConstant(conv2d_call1->args[5]);

    // Calculate requantization multiplier and shift
    Device dev{DLDeviceType::kDLCPU, 0};
    runtime::NDArray multiplier1_nda =
        runtime::NDArray::Empty({out_channels1}, DataType::Int(32), dev);
    runtime::NDArray shift1_nda = runtime::NDArray::Empty({out_channels1}, DataType::Int(32), dev);
    int32_t* multiplier1 = static_cast<int32_t*>(multiplier1_nda->data);
    int32_t* shift1 = static_cast<int32_t*>(shift1_nda->data);
    for (int i = 0; i < out_channels1; ++i) {
      double effective_output_scale1 = static_cast<double>(input_scale1) * filter_scales1[i] /
                                       static_cast<double>(output_scale1);
      std::tie(*(multiplier1 + i), *(shift1 + i)) =
          tvm::relay::qnn::GetFixedPointMultiplierShift(effective_output_scale1);
    }

    int64_t out_channels2 = conv2d_attrs2->channels.as<IntImmNode>()->value;
    float output_scale2 = GetScalarFromConstant<float>(requantize_call2->args[3]);
    // output_scale2 = GetScalarFromConstant<float>(clip_call2->args[3]);

    auto input_scale2 = GetScalarFromConstant<float>(conv2d_call2->args[4]);
    auto filter_scales2 = tvm::relay::qnn::GetFloatVectorFromConstant(conv2d_call2->args[5]);

    runtime::NDArray multiplier2_nda =
        runtime::NDArray::Empty({out_channels2}, DataType::Int(32), dev);
    runtime::NDArray shift2_nda = runtime::NDArray::Empty({out_channels2}, DataType::Int(32), dev);
    int32_t* multiplier2 = static_cast<int32_t*>(multiplier2_nda->data);
    int32_t* shift2 = static_cast<int32_t*>(shift2_nda->data);
    for (int i = 0; i < out_channels2; ++i) {
      double effective_output_scale2 = static_cast<double>(input_scale2) * filter_scales2[i] /
                                       static_cast<double>(output_scale2);
      std::tie(*(multiplier2 + i), *(shift2 + i)) =
          tvm::relay::qnn::GetFixedPointMultiplierShift(effective_output_scale2);
    }

    // Create constants from requantization multiplier and shift
    Constant multiplier1_const(multiplier1_nda);
    Constant shift1_const(shift1_nda);
    Constant multiplier2_const(multiplier2_nda);
    Constant shift2_const(shift2_nda);

    // Convert scale scalars into Constants
    // Scales are expected as Constants by following passes
    Expr weight1_scale = conv2d_call1->args[5];
    Expr req1_inp_scale = requantize_call1->args[1];
    if (out_channels1 == 1) {
      runtime::NDArray weight1_scale_nda =
          runtime::NDArray::Empty({out_channels1}, DataType::Float(32), dev);
      float* weight1_scale_p = static_cast<float*>(weight1_scale_nda->data);
      *weight1_scale_p = GetScalarFromConstant<float>(weight1_scale);
      weight1_scale = Constant(weight1_scale_nda);

      runtime::NDArray req1_inp_scale_nda =
          runtime::NDArray::Empty({out_channels1}, DataType::Float(32), dev);
      float* req1_inp_scale_p = static_cast<float*>(req1_inp_scale_nda->data);
      *req1_inp_scale_p = GetScalarFromConstant<float>(req1_inp_scale);
      req1_inp_scale = Constant(req1_inp_scale_nda);
    }

    Expr weight2_scale = conv2d_call2->args[5];
    Expr req2_inp_scale = requantize_call2->args[1];
    if (out_channels2 == 1) {
      runtime::NDArray weight2_scale_nda =
          runtime::NDArray::Empty({out_channels2}, DataType::Float(32), dev);
      float* weight2_scale_p = static_cast<float*>(weight2_scale_nda->data);
      *weight2_scale_p = GetScalarFromConstant<float>(weight2_scale);
      weight2_scale = Constant(weight2_scale_nda);

      runtime::NDArray req2_inp_scale_nda =
          runtime::NDArray::Empty({out_channels2}, DataType::Float(32), dev);
      float* req2_inp_scale_p = static_cast<float*>(req2_inp_scale_nda->data);
      *req2_inp_scale_p = GetScalarFromConstant<float>(req2_inp_scale);
      req2_inp_scale = Constant(req2_inp_scale_nda);
    }

    // Replace existing weights (HWIO) with the transposed ones (OHWI) for Conv2D
    // Substitute Conv2D weight_zero_point with the CMSIS-NN multiplier
    // Substitute Requantize input_zero_point with CMSIS-NN shift
    // Conv2D arguments: data, weight, input_zp, weight_zp, input_sc, weight_sc
    Call ret_call;
    Array<Expr> conv2d_args1;
    if (pad_call1) {
      ret_call =
          Call(pad_call1->op, {pad_call1->args[0], pad_call1->args[1]}, pad_call1->attrs, {});
      conv2d_args1 = {ret_call,          conv2d_kernel1,        conv2d_call1->args[2],
                      multiplier1_const, conv2d_call1->args[4], weight1_scale};
    } else
      conv2d_args1 = {conv2d_call1->args[0], conv2d_kernel1,        conv2d_call1->args[2],
                      multiplier1_const,     conv2d_call1->args[4], weight1_scale};
    ret_call = Call(conv2d_call1->op, conv2d_args1, new_conv2d_attrs1, {});
    if (bias_add_call1)
      ret_call =
          Call(bias_add_call1->op, {ret_call, bias_add_call1->args[1]}, bias_add_call1->attrs, {});
    Array<Expr> requantize_args1 = {ret_call, req1_inp_scale, shift1_const,
                                    requantize_call1->args[3], requantize_call1->args[4]};
    ret_call = Call(requantize_call1->op, requantize_args1, requantize_call1->attrs, {});
    if (clip_call1) ret_call = Call(clip_call1->op, {ret_call}, clip_call1->attrs, {});
    if (pad_call2)
      ret_call = Call(pad_call2->op, {ret_call, pad_call2->args[1]}, pad_call2->attrs, {});
    Array<Expr> conv2d_args2 = {ret_call,          conv2d_kernel2,        conv2d_call2->args[2],
                                multiplier2_const, conv2d_call2->args[4], weight2_scale};
    ret_call = Call(conv2d_call2->op, conv2d_args2, new_conv2d_attrs2, {});
    if (bias_add_call2)
      ret_call =
          Call(bias_add_call2->op, {ret_call, bias_add_call2->args[1]}, bias_add_call2->attrs, {});
    Array<Expr> requantize_args2 = {ret_call, req2_inp_scale, shift2_const,
                                    requantize_call2->args[3], requantize_call2->args[4]};
    ret_call = Call(requantize_call2->op, requantize_args2, requantize_call2->attrs, {});
    if (clip_call2) ret_call = Call(clip_call2->op, {ret_call}, clip_call2->attrs, {});

    return std::move(ret_call);
  }

  Expr GenerateMul2RequantConstants(const Expr& expr) {
    auto* final_call = expr.as<CallNode>();
    auto* final_op = final_call->op.as<OpNode>();
    const CallNode* mul1_call = nullptr;
    const CallNode* mul2_call = nullptr;

    if (final_op->name == "qnn.requantize")
      mul1_call = final_call->args[0].as<CallNode>();
    else
      mul1_call = final_call;

    mul2_call = mul1_call->args[0].as<CallNode>();

    Array<PrimExpr> input_shape = mul2_call->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> m2_shape = mul2_call->args[1]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> m1_shape = mul1_call->args[1]->type_as<TensorTypeNode>()->shape;

    int32_t input_size = 1, m2_size = 1, m1_size = 1;
    size_t i;
    std::vector<int64_t> in_shape;
    for (i = 0; i < input_shape.size(); i++) {
      in_shape.push_back(input_shape[i].as<tvm::IntImmNode>()->value);
      input_size *= input_shape[i].as<tvm::IntImmNode>()->value;
    }

    for (i = 0; i < m2_shape.size(); i++) m2_size *= m2_shape[i].as<tvm::IntImmNode>()->value;
    for (i = 0; i < m1_shape.size(); i++) m1_size *= m1_shape[i].as<tvm::IntImmNode>()->value;

    Device dev{DLDeviceType::kDLCPU, 0};
    runtime::NDArray Const_mul = runtime::NDArray::Empty(in_shape, DataType::Int(8), dev);
    int8_t* Const_mul_data = static_cast<int8_t*>(Const_mul->data);
    int8_t* m2_const = static_cast<int8_t*>(mul2_call->args[1].as<ConstantNode>()->data->data);
    int8_t* m1_const = static_cast<int8_t*>(mul1_call->args[1].as<ConstantNode>()->data->data);

    for (int i = 0; i < input_size; i++)
      *(Const_mul_data + i) = m2_const[i % m2_size] * m1_const[i % m1_size] / 127;

    float m2_scale = GetScalarFromConstant<float>(mul2_call->args[4]);
    float m1_scale = GetScalarFromConstant<float>(mul1_call->args[4]);
    runtime::NDArray m_scale = runtime::NDArray::Empty({}, DataType::Float(32), dev);
    float* m_scale_data = static_cast<float*>(m_scale->data);

    *m_scale_data = m2_scale * m1_scale * 127;

    Constant Const_merged(Const_mul);
    Constant m_scale_new(m_scale);

    Call ret_call;
    if (final_op->name == "qnn.requantize")
      ret_call = Call(mul1_call->op,
                      {mul2_call->args[0], Const_merged, mul2_call->args[2], mul2_call->args[3],
                       m_scale_new, mul2_call->args[5], final_call->args[3], final_call->args[4]},
                      mul1_call->attrs, {});
    else
      ret_call = Call(mul1_call->op,
                      {mul2_call->args[0], Const_merged, mul2_call->args[2], mul2_call->args[3],
                       m_scale_new, mul2_call->args[5], mul1_call->args[6], mul1_call->args[7]},
                      mul1_call->attrs, {});
    return std::move(ret_call);
  }

  Expr GenerateMAMMRequantConstants(const Expr& expr) {
    auto* final_call = expr.as<CallNode>();
    auto* final_op = final_call->op.as<OpNode>();
    const CallNode* mul1_call = nullptr;
    const CallNode* mul2_call = nullptr;
    const CallNode* add_call = nullptr;
    const CallNode* mul3_call = nullptr;

    if (final_op->name == "qnn.requantize")
      mul1_call = final_call->args[0].as<CallNode>();
    else
      mul1_call = final_call;

    mul2_call = mul1_call->args[0].as<CallNode>();
    add_call = mul2_call->args[0].as<CallNode>();
    mul3_call = add_call->args[0].as<CallNode>();

    Array<PrimExpr> input_shape = mul3_call->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> m3_shape = mul3_call->args[1]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> a_shape = add_call->args[1]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> m2_shape = mul2_call->args[1]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> m1_shape = mul1_call->args[1]->type_as<TensorTypeNode>()->shape;

    int32_t input_size = 1, m3_size = 1, a_size = 1, m2_size = 1, m1_size = 1;
    size_t i;
    std::vector<int64_t> in_shape;
    for (i = 0; i < input_shape.size(); i++) {
      in_shape.push_back(input_shape[i].as<tvm::IntImmNode>()->value);
      input_size *= input_shape[i].as<tvm::IntImmNode>()->value;
    }

    for (i = 0; i < m3_shape.size(); i++) m3_size *= m3_shape[i].as<tvm::IntImmNode>()->value;
    for (i = 0; i < a_shape.size(); i++) a_size *= a_shape[i].as<tvm::IntImmNode>()->value;
    for (i = 0; i < m2_shape.size(); i++) m2_size *= m2_shape[i].as<tvm::IntImmNode>()->value;
    for (i = 0; i < m1_shape.size(); i++) m1_size *= m1_shape[i].as<tvm::IntImmNode>()->value;

    Device dev{DLDeviceType::kDLCPU, 0};
    runtime::NDArray Const_mul = runtime::NDArray::Empty(in_shape, DataType::Int(8), dev);
    runtime::NDArray Const_add = runtime::NDArray::Empty(in_shape, DataType::Int(8), dev);
    std::vector<float> Const_tmp1;
    std::vector<float> Const_tmp2;
    int8_t* Const_mul_data = static_cast<int8_t*>(Const_mul->data);
    int8_t* Const_add_data = static_cast<int8_t*>(Const_add->data);
    int8_t* m3_const = static_cast<int8_t*>(mul3_call->args[1].as<ConstantNode>()->data->data);
    int8_t* a_const = static_cast<int8_t*>(add_call->args[1].as<ConstantNode>()->data->data);
    int8_t* m2_const = static_cast<int8_t*>(mul2_call->args[1].as<ConstantNode>()->data->data);
    int8_t* m1_const = static_cast<int8_t*>(mul1_call->args[1].as<ConstantNode>()->data->data);

    float m3_scale = GetScalarFromConstant<float>(mul3_call->args[4]);
    float a_scale = GetScalarFromConstant<float>(add_call->args[4]);
    float m2_scale = GetScalarFromConstant<float>(mul2_call->args[4]);
    float m1_scale = GetScalarFromConstant<float>(mul1_call->args[4]);
    runtime::NDArray m_scale = runtime::NDArray::Empty({}, DataType::Float(32), dev);
    runtime::NDArray add_scale = runtime::NDArray::Empty({}, DataType::Float(32), dev);
    float* m_scale_data = static_cast<float*>(m_scale->data);
    float* add_scale_data = static_cast<float*>(add_scale->data);

    for (int i = 0; i < input_size; i++) {
      Const_tmp1.push_back(m3_const[i % m3_size] * m2_const[i % m2_size] * m1_const[i % m1_size] *
                           m3_scale * m2_scale * m1_scale);
      Const_tmp2.push_back(a_const[i % a_shape[3].as<tvm::IntImmNode>()->value +
                                   (i / (input_shape[2].as<tvm::IntImmNode>()->value *
                                         input_shape[3].as<tvm::IntImmNode>()->value)) *
                                       a_shape[3].as<tvm::IntImmNode>()->value] *
                           m2_const[i % m2_size] * m1_const[i % m1_size] * a_scale * m2_scale *
                           m1_scale);
    }

    float Const_mul_max = (std::abs(*max_element(Const_tmp1.begin(), Const_tmp1.end())) >
                           std::abs(*min_element(Const_tmp1.begin(), Const_tmp1.end())))
                              ? std::abs(*max_element(Const_tmp1.begin(), Const_tmp1.end()))
                              : std::abs(*min_element(Const_tmp1.begin(), Const_tmp1.end()));
    float Const_add_max = (std::abs(*max_element(Const_tmp2.begin(), Const_tmp2.end())) >
                           std::abs(*min_element(Const_tmp2.begin(), Const_tmp2.end())))
                              ? std::abs(*max_element(Const_tmp2.begin(), Const_tmp2.end()))
                              : std::abs(*min_element(Const_tmp2.begin(), Const_tmp2.end()));

    for (int i = 0; i < input_size; i++) {
      *(Const_mul_data + i) = std::round(127.0 * Const_tmp1[i] / Const_mul_max);
      *(Const_add_data + i) = std::round(127.0 * Const_tmp2[i] / Const_add_max);
    }

    *m_scale_data = Const_mul_max / 127.0;
    *add_scale_data = Const_add_max / 127.0;

    Constant Const_merged_m(Const_mul);
    Constant m_scale_new(m_scale);
    Constant Const_merged_a(Const_add);
    Constant add_scale_new(add_scale);

    Call ret_call;
    if (final_op->name == "qnn.requantize") {
      ret_call = Call(mul3_call->op,
                      {mul3_call->args[0], Const_merged_m, mul3_call->args[2], mul3_call->args[3],
                       m_scale_new, mul3_call->args[5], add_call->args[2], add_call->args[3]},
                      mul3_call->attrs, {});
      ret_call = Call(add_call->op,
                      {ret_call, Const_merged_a, add_call->args[2], add_call->args[3],
                       add_scale_new, add_call->args[5], final_call->args[3], final_call->args[4]},
                      add_call->attrs, {});
    } else {
      ret_call = Call(mul3_call->op,
                      {mul3_call->args[0], Const_merged_m, mul3_call->args[2], mul3_call->args[3],
                       m_scale_new, mul3_call->args[5], add_call->args[2], add_call->args[3]},
                      mul3_call->attrs, {});
      ret_call = Call(add_call->op,
                      {ret_call, Const_merged_a, add_call->args[2], add_call->args[3],
                       add_scale_new, add_call->args[5], mul1_call->args[6], mul1_call->args[7]},
                      add_call->attrs, {});
    }
    return std::move(ret_call);
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Expr final_call = post;
    auto* post_call = post.as<CallNode>();

    auto* global_var = call->op.as<GlobalVarNode>();
    if (global_var) {
      // Update to global function call needed because the body changes while
      // generating new constants
      Function func = Downcast<Function>(mod_->Lookup(global_var->name_hint));
      Expr new_body = VisitExpr(func->body);
      if (!new_body.same_as(func->body)) {
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        mod_->Update(GetRef<GlobalVar>(global_var), new_func);
        final_call = Call(GetRef<GlobalVar>(global_var), post_call->args);
      }
    }

    // Recreate composite function and corresponding call
    // Updated composite function contains CMSIS-NN quantized multiplier and shift constants
    if (call->op.as<FunctionNode>()) {
      auto* func = call->op.as<FunctionNode>();
      auto func_name = func->GetAttr<String>(attr::kComposite);
      if (func_name.defined() && func_name == "cmsis-nn.qnn_conv2d") {
        Expr new_body = GenerateConv2dRequantConstants(func->body);
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        final_call = Call(new_func, post_call->args);
      } else if (func_name.defined() && func_name == "cmsis-nn.fiti_reduce_transaction1") {
        Expr new_body = GenerateConv2dRequantConstants_1(func->body);
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        final_call = Call(new_func, post_call->args);
      } else if (func_name.defined() && func_name == "cmsis-nn.fiti_reduce_transaction2") {
        Expr new_body = GenerateConv2dRequantConstants_2(func->body);
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        final_call = Call(new_func, post_call->args);
      } else if (func_name.defined() && func_name == "cmsis-nn.qnn_mul2") {
        Expr new_body = GenerateMul2RequantConstants(func->body);
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        final_call = Call(new_func, post_call->args);
      } else if (func_name.defined() && func_name == "cmsis-nn.fiti_mamm") {
        Expr new_body = GenerateMAMMRequantConstants(func->body);
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        final_call = Call(new_func, post_call->args);
      }
    }

    final_call->span = call->span;
    return final_call;
  }

 private:
  IRModule mod_;
};

IRModule GenerateConstants(const IRModule& mod) {
  String func_name;
  Function func;

  // Introduces CMSIS-NN constants before the call to the external Relay function
  auto generate_constants = GenerateConstantsMutator(mod);
  Function main_func = Downcast<Function>(mod->Lookup("main"));
  auto new_main_body = generate_constants.VisitExpr(main_func->body);
  if (!new_main_body.same_as(main_func->body)) {
    auto main_var = mod->GetGlobalVar("main");
    auto new_main_func = Function(main_func->params, new_main_body, main_func->ret_type,
                                  main_func->type_params, main_func->attrs);
    mod->Update(main_var, new_main_func);
  }

  return mod;
}

transform::Pass GenerateCMSISNNConstants() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule m, transform::PassContext pc) { return GenerateConstants(m); };
  return tvm::transform::CreateModulePass(pass_func, 0, "GenerateCMSISNNConstants", {});
}

TVM_REGISTER_GLOBAL("relay.ext.cmsisnn.transform.GenerateCMSISNNConstants")
    .set_body_typed(GenerateCMSISNNConstants);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
