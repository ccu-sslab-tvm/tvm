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

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../../../runtime/file_utils.h"
#include "../../../../target/source/codegen_c.h"
#include "../../../../target/source/codegen_c_host.h"
#include "compiler_attrs.h"

namespace tvm {
using namespace tir;
namespace relay {
namespace contrib {
namespace cmsisnn {

class CodeGenCMSISNN : public codegen::CodeGenCHost {
 public:
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl, std::string target_str,
            bool debug_last_error) {
    this->debug_last_error = debug_last_error;
    std::unordered_set<std::string> devices;
    devices.insert("cmsis-nn");
    CodeGenCHost::Init(output_ssa, emit_asserts, emit_fwd_func_decl, target_str, devices);
  }

  /*!
   * \brief Emit code that offloads a subgraph to the Cortex-M
   *
   * \return string of code that offloads a subgraph to the Cortex-M
   */
  void AddFunction(const PrimFunc& prim_func) { CodeGenC::AddFunction(prim_func); }

 private:
  /*!  * \brief Enable storing the last error */
  bool debug_last_error;

  /*!  * \brief CMSIS-NN context buffer info */
  struct CMSISNNContextBuffer {
    std::string name;
    int size;
  };

  /*!  * \brief CMSIS-NN buffer dimensions */
  struct CMSISNNDims {
    int n;
    int h;
    int w;
    int c;
  };

  /*!  * \brief CMSIS-NN Conv2D and Depthwise parameters */
  struct Conv2DParams {
    int input_offset;
    int output_offset;
    int stride_w;
    int stride_h;
    int padding_w;
    int padding_h;
    int dilation_w;
    int dilation_h;
    int clip_min;
    int clip_max;
    int depth_multiplier;
  };

  /*!  * \brief CMSIS-NN Conv2D and Depthwise parameters */
  struct FCParams {
    int input_offset;
    int filter_offset;
    int output_offset;
    int clip_min;
    int clip_max;
    int multiplier;
    int shift;
  };

  struct PoolParams {
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int clip_min;
    int clip_max;
  };

  using codegen::CodeGenCHost::VisitStmt_;

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern */
  void VisitExpr_(const CallNode* op, std::ostream& os) final {
    if (!op->op.same_as(builtin::call_extern())) {
      CodeGenCHost::VisitExpr_(op, os);
      return;
    }
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;
    if (cmsis_func_name == "arm_softmax_s8" || cmsis_func_name == "arm_elementwise_mul_s8" ||
        cmsis_func_name == "arm_elementwise_add_s8" || cmsis_func_name == "fiti_sigmoid" || cmsis_func_name == "fiti_resize2d" || cmsis_func_name == "fiti_concat" || 
        cmsis_func_name == "fiti_requant" || cmsis_func_name == "fiti_add" || cmsis_func_name == "fiti_mul" || cmsis_func_name == "fiti_max_pool" || cmsis_func_name == "fiti_slice") {
      CodeGenC::VisitExpr_(op, os);
    } else if (cmsis_func_name == "arm_convolve_wrapper_s8" ||
               cmsis_func_name == "arm_convolve_wrapper_s16" ||
               cmsis_func_name == "arm_depthwise_conv_wrapper_s8" ||
               cmsis_func_name == "arm_depthwise_conv_wrapper_s16") {
      EmitConv2D(op);
    } 
    else if (cmsis_func_name == "fiti_convolve_wrapper_s82")
      FITI_reduce_transaction2(op);
    else if (cmsis_func_name == "fiti_convolve_wrapper_s83")
      FITI_reduce_transaction1(op);
    else if(cmsis_func_name == "fiti_mamm")
      FITI_MAMM(op);
    else if (cmsis_func_name == "arm_fully_connected_s8" ||
               cmsis_func_name == "arm_fully_connected_s16") {
      EmitFullyConnected(op);
    } else if (cmsis_func_name == "arm_avgpool_s8" || cmsis_func_name == "arm_avgpool_s16" ||
               cmsis_func_name == "arm_max_pool_s8" || cmsis_func_name == "arm_max_pool_s16") {
      EmitPool2D(op);
    }
    return;
  }

  /*!  * \brief Emits cmsis_nn_context struct */
  std::string EmitCMSISNNContext(std::ostream& os, CMSISNNContextBuffer context_buffer) {
    std::string struct_name = "context";
    PrintIndent();
    os << "cmsis_nn_context " << struct_name << "= {" << context_buffer.name << ","
       << context_buffer.size << "};\n";
    return struct_name;
  }

  std::string FITIConvParams(std::ostream& os, Conv2DParams params, std::string no) {
    std::string struct_name = "cmsis_nn_conv_params";
    std::string instance_name = "conv_params";
    instance_name += no;
    if (params.depth_multiplier != -1) {
      struct_name = "cmsis_nn_dw_conv_params";
    }
    PrintIndent();
    os << "cmsis_nn_tile stride" << no << " = {" << params.stride_w << "," << params.stride_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile padding" << no << " = {" << params.padding_w << "," << params.padding_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile dilation" << no << " = {" << params.dilation_w << "," << params.dilation_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_activation activation" << no << " = {" << params.clip_min << "," << params.clip_max
       << "};\n";
    PrintIndent();
    os << struct_name << " " << instance_name << " = {" << params.input_offset << ", "
       << params.output_offset;
    if (params.depth_multiplier != -1) {
      os << ", " << params.depth_multiplier;
    }
    os << ", stride" << no << ", padding" << no << ", dilation" << no << ", activation" << no << "};\n";
    return instance_name;
  }

  /*!  * \brief Emits cmsis_nn_conv_params struct */
  std::string EmitCMSISNNConvParams(std::ostream& os, Conv2DParams params) {
    std::string struct_name = "cmsis_nn_conv_params";
    std::string instance_name = "conv_params";
    if (params.depth_multiplier != -1) {
      struct_name = "cmsis_nn_dw_conv_params";
    }
    PrintIndent();
    os << "cmsis_nn_tile stride = {" << params.stride_w << "," << params.stride_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile padding = {" << params.padding_w << "," << params.padding_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile dilation = {" << params.dilation_w << "," << params.dilation_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_activation activation = {" << params.clip_min << "," << params.clip_max
       << "};\n";
    PrintIndent();
    os << struct_name << " " << instance_name << " = {" << params.input_offset << ", "
       << params.output_offset;
    if (params.depth_multiplier != -1) {
      os << ", " << params.depth_multiplier;
    }
    os << ", stride, padding, dilation, activation};\n";
    return instance_name;
  }

  /*!  * \brief Emits cmsis_nn_fc_params struct */
  std::string EmitCMSISNNFCParams(std::ostream& os, FCParams params) {
    std::string struct_name = "cmsis_nn_fc_params";
    std::string instance_name = "fc_params";
    PrintIndent();
    os << "cmsis_nn_activation activation = {" << params.clip_min << "," << params.clip_max
       << "};\n";
    PrintIndent();
    os << struct_name << " " << instance_name << " = {" << params.input_offset << ", "
       << params.filter_offset << ", " << params.output_offset;
    os << ", activation};\n";
    return instance_name;
  }

  /*!  * \brief Emits cmsis_nn_pool_params struct */
  std::string EmitCMSISNNPoolParams(std::ostream& os, PoolParams params) {
    std::string struct_name = "cmsis_nn_pool_params";
    std::string instance_name = "pool_params";
    PrintIndent();
    os << "cmsis_nn_tile stride = {" << params.stride_w << "," << params.stride_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_tile padding = {" << params.padding_w << "," << params.padding_h << "};\n";
    PrintIndent();
    os << "cmsis_nn_activation activation = {" << params.clip_min << "," << params.clip_max
       << "};\n";
    PrintIndent();
    os << struct_name << " " << instance_name << " = {stride, padding, activation};\n";
    return instance_name;
  }

  std::string FITIQuantParams(std::ostream& os, std::string multiplier, std::string shift, std::string no) 
  {
    std::string struct_name = "quant_params";
    struct_name += no;
    PrintIndent();
    os << "cmsis_nn_per_channel_quant_params " << struct_name << " = {" << multiplier << ", " << shift << "};\n";
    return struct_name;
  }

  /*!  * \brief Emits cmsis_nn_per_channel_quant_params struct */
  std::string EmitCMSISNNPerChannelQuantParams(std::ostream& os, std::string multiplier,
                                               std::string shift) {
    std::string struct_name = "quant_params";
    PrintIndent();
    os << "cmsis_nn_per_channel_quant_params " << struct_name << " = {" << multiplier << ", "
       << shift << "};\n";
    return struct_name;
  }

  /*!  * \brief Emits cmsis_nn_per_tensor_quant_params struct */
  std::string EmitCMSISNNPerTensorQuantParams(std::ostream& os, int multiplier, int shift) {
    std::string struct_name = "quant_params";
    PrintIndent();
    os << "cmsis_nn_per_tensor_quant_params " << struct_name << " = {" << multiplier << ", "
       << shift << "};\n";
    return struct_name;
  }

  /*!  * \brief Emits cmsis_nn_dims struct */
  std::string EmitCMSISNNDims(std::ostream& os, std::string tensor_type, CMSISNNDims dims) {
    std::string struct_name = tensor_type + "_dims";
    PrintIndent();
    os << "cmsis_nn_dims " << struct_name << " = {" << dims.n << "," << dims.h << "," << dims.w
       << "," << dims.c << "};\n";
    return struct_name;
  }

  /*!  * \brief Deduces variable name from call_extern argument resting at id */
  std::string VarNameFromArg(const CallNode* op, int id) {
    return op->args[id].as<VarNode>()->name_hint.c_str();
  }

  /*!  * \brief Deduces value from call_extern argument resting at id */
  int ValueFromArg(const CallNode* op, int id) { return op->args[id].as<IntImmNode>()->value; }

  /*!  * \brief extracts CMSIS-NN context buffer information */
  CMSISNNContextBuffer extract_context_buffer_info(const CallNode* op, int base_pos) {
    CMSISNNContextBuffer context_buffer;

    // The argument could be a Var if it is allocated to hold the
    // context buffer OR it will be a StringImm with "NULL"
    if (op->args[base_pos]->IsInstance<VarNode>()) {
      context_buffer.name = op->args[base_pos].as<VarNode>()->name_hint;
    } else {
      context_buffer.name = op->args[base_pos].as<StringImmNode>()->value;
    }
    context_buffer.size = ValueFromArg(op, base_pos + 1);
    return context_buffer;
  }

  /*!  * \brief extracts CMSIS-NN conv2d parameters from call_extern */
  Conv2DParams extract_conv2d_params(const CallNode* op, int base_pos) {
    Conv2DParams conv2d_params;
    conv2d_params.input_offset = ValueFromArg(op, base_pos);
    conv2d_params.output_offset = ValueFromArg(op, ++base_pos);
    conv2d_params.stride_w = ValueFromArg(op, ++base_pos);
    conv2d_params.stride_h = ValueFromArg(op, ++base_pos);
    conv2d_params.padding_w = ValueFromArg(op, ++base_pos);
    conv2d_params.padding_h = ValueFromArg(op, ++base_pos);
    conv2d_params.dilation_w = ValueFromArg(op, ++base_pos);
    conv2d_params.dilation_h = ValueFromArg(op, ++base_pos);
    conv2d_params.clip_min = ValueFromArg(op, ++base_pos);
    conv2d_params.clip_max = ValueFromArg(op, ++base_pos);
    conv2d_params.depth_multiplier = ValueFromArg(op, ++base_pos);
    return conv2d_params;
  }

  /*!  * \brief extracts CMSIS-NN FC parameters from call_extern */
  FCParams extract_fc_params(const CallNode* op, int base_pos) {
    FCParams fc_params;
    fc_params.input_offset = ValueFromArg(op, base_pos);
    fc_params.filter_offset = ValueFromArg(op, ++base_pos);
    fc_params.output_offset = ValueFromArg(op, ++base_pos);
    fc_params.clip_min = ValueFromArg(op, ++base_pos);
    fc_params.clip_max = ValueFromArg(op, ++base_pos);
    fc_params.multiplier = ValueFromArg(op, ++base_pos);
    fc_params.shift = ValueFromArg(op, ++base_pos);
    return fc_params;
  }

  /*!  * \brief extracts CMSIS-NN Pooling parameters from call_extern */
  PoolParams extract_pool_params(const CallNode* op, int base_pos) {
    PoolParams pool_params;
    pool_params.stride_h = ValueFromArg(op, base_pos);
    pool_params.stride_w = ValueFromArg(op, ++base_pos);
    pool_params.padding_h = ValueFromArg(op, ++base_pos);
    pool_params.padding_w = ValueFromArg(op, ++base_pos);
    pool_params.clip_min = ValueFromArg(op, ++base_pos);
    pool_params.clip_max = ValueFromArg(op, ++base_pos);
    return pool_params;
  }

  /*!  * \brief extracts CMSIS-NN buffer dimensions from call_extern */
  CMSISNNDims extract_buffer_dims(const CallNode* op, int base_pos) {
    CMSISNNDims dims;
    dims.n = ValueFromArg(op, base_pos);
    dims.h = ValueFromArg(op, ++base_pos);
    dims.w = ValueFromArg(op, ++base_pos);
    dims.c = ValueFromArg(op, ++base_pos);
    return dims;
  }

  void FITI_MAMM(const CallNode* op)
  {
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;
    // extract buffer names from call_extern
    int arg_id = 0;
    std::string input_data = VarNameFromArg(op, ++arg_id);
    std::string Const_mul = VarNameFromArg(op, ++arg_id);
    std::string Const_add = VarNameFromArg(op, ++arg_id);
    std::string output = VarNameFromArg(op, ++arg_id);
    int mul_input_zp = ValueFromArg(op, ++arg_id);
    int mul_const_zp = ValueFromArg(op, ++arg_id);
    int mul_output_zp = ValueFromArg(op, ++arg_id);
    int mul_multiplier = ValueFromArg(op, ++arg_id);
    int mul_shift = ValueFromArg(op, ++arg_id);
    int add_input_zp = ValueFromArg(op, ++arg_id);
    int add_in_multiplier = ValueFromArg(op, ++arg_id);
    int add_in_shift = ValueFromArg(op, ++arg_id);
    int add_const_zp = ValueFromArg(op, ++arg_id);
    int add_const_multiplier = ValueFromArg(op, ++arg_id);
    int add_const_shift = ValueFromArg(op, ++arg_id);
    int add_output_zp = ValueFromArg(op, ++arg_id);
    int add_out_multiplier = ValueFromArg(op, ++arg_id);
    int add_out_shift = ValueFromArg(op, ++arg_id);
    int size = ValueFromArg(op, ++arg_id);

    // Emit CMSIS-NN API
    PrintIndent();
    //stream << cmsis_func_name << "(";
    stream << "fiti_mul(";
    stream << input_data << ", ";
    stream << Const_mul << ", ";
    stream << mul_input_zp << ", ";
    stream << "1073741824, 1, ";
    stream << mul_const_zp << ", ";
    stream << "1073741824, 1, ";
    stream << output << ", ";
    stream << mul_output_zp << ", ";
    stream << mul_multiplier << ", ";
    stream << mul_shift << ", ";
    stream << "-128, 127, ";
    stream << size << ");\n";

    PrintIndent();
    //stream << cmsis_func_name << "(";
    stream << "fiti_add(";
    stream << output << ", ";
    stream << Const_add << ", ";
    stream << add_input_zp << ", ";
    stream << add_in_multiplier << ", ";
    stream << add_in_shift << ", ";
    stream << add_const_zp << ", ";
    stream << add_const_multiplier << ", ";
    stream << add_const_shift << ", ";
    stream << output << ", ";
    stream << add_output_zp << ", ";
    stream << add_out_multiplier << ", ";
    stream << add_out_shift << ", ";
    stream << "-128, 127, ";
    stream << size << ");\n";
  }

  void FITI_reduce_transaction1(const CallNode* op) 
  {
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;
    // extract buffer names from call_extern
    int arg_id = 0;
    
    std::string input_data1 = VarNameFromArg(op, ++arg_id);
    std::string filter_data1 = VarNameFromArg(op, ++arg_id);
    std::string multiplier1 = VarNameFromArg(op, ++arg_id);
    std::string bias_data1 = VarNameFromArg(op, ++arg_id);
    std::string shift1 = VarNameFromArg(op, ++arg_id);
    //std::string output_data1 = VarNameFromArg(op, ++arg_id);
    //std::string input_data2 = VarNameFromArg(op, ++arg_id);
    std::string filter_data2 = VarNameFromArg(op, ++arg_id);
    std::string multiplier2 = VarNameFromArg(op, ++arg_id);
    std::string bias_data2 = VarNameFromArg(op, ++arg_id);
    std::string shift2 = VarNameFromArg(op, ++arg_id);
    //std::string output_data2 = VarNameFromArg(op, ++arg_id);
    //std::string input_data3 = VarNameFromArg(op, ++arg_id);
    std::string filter_data3 = VarNameFromArg(op, ++arg_id);
    std::string multiplier3 = VarNameFromArg(op, ++arg_id);
    std::string bias_data3 = VarNameFromArg(op, ++arg_id);
    std::string shift3 = VarNameFromArg(op, ++arg_id);
    std::string output_data3 = VarNameFromArg(op, ++arg_id);

    Conv2DParams conv2d_params1 = extract_conv2d_params(op, ++arg_id);
    Conv2DParams conv2d_params2 = extract_conv2d_params(op, arg_id+11);
    Conv2DParams conv2d_params3 = extract_conv2d_params(op, arg_id+22);
    CMSISNNDims input_dims1 = extract_buffer_dims(op, arg_id+33);
    CMSISNNDims filter_dims1 = extract_buffer_dims(op, arg_id+37);
    CMSISNNDims bias_dims1 = extract_buffer_dims(op, arg_id+41);
    CMSISNNDims output_dims1 = extract_buffer_dims(op, arg_id+45);
    CMSISNNDims input_dims2 = extract_buffer_dims(op, arg_id+49);
    CMSISNNDims filter_dims2 = extract_buffer_dims(op, arg_id+53);
    CMSISNNDims bias_dims2 = extract_buffer_dims(op, arg_id+57);
    CMSISNNDims output_dims2 = extract_buffer_dims(op, arg_id+61);
    CMSISNNDims input_dims3 = extract_buffer_dims(op, arg_id+65);
    CMSISNNDims filter_dims3 = extract_buffer_dims(op, arg_id+69);
    CMSISNNDims bias_dims3 = extract_buffer_dims(op, arg_id+73);
    CMSISNNDims output_dims3 = extract_buffer_dims(op, arg_id+77);

    if(filter_dims1.c != input_dims1.c)
    {
      filter_dims1.h = std::sqrt(filter_dims1.c / input_dims1.c);
      filter_dims1.w = std::sqrt(filter_dims1.c / input_dims1.c);
      filter_dims1.c = input_dims1.c;
    }

    if(filter_dims2.c != input_dims2.c)
    {
      filter_dims2.h = std::sqrt(filter_dims2.c / input_dims2.c);
      filter_dims2.w = std::sqrt(filter_dims2.c / input_dims2.c);
      filter_dims2.c = input_dims2.c;
    }

    if(filter_dims3.c != input_dims3.c)
    {
      filter_dims3.h = std::sqrt(filter_dims3.c / input_dims3.c);
      filter_dims3.w = std::sqrt(filter_dims3.c / input_dims3.c);
      filter_dims3.c = input_dims3.c;
    }

    // Emit CMSIS-NN API arguments
    std::string conv_params1 = FITIConvParams(stream, conv2d_params1, "1");
    std::string quant_params1 = FITIQuantParams(stream, multiplier1, shift1, "1");
    std::string input_dim1 = EmitCMSISNNDims(stream, "input1", input_dims1);
    std::string filter_dim1 = EmitCMSISNNDims(stream, "filter1", filter_dims1);
    std::string bias_dim1 = EmitCMSISNNDims(stream, "bias1", bias_dims1);
    std::string output_dim1 = EmitCMSISNNDims(stream, "output1", output_dims1);

    std::string conv_params2 = FITIConvParams(stream, conv2d_params2, "2");
    std::string quant_params2 = FITIQuantParams(stream, multiplier2, shift2, "2");
    std::string input_dim2 = EmitCMSISNNDims(stream, "input2", input_dims2);
    std::string filter_dim2 = EmitCMSISNNDims(stream, "filter2", filter_dims2);
    std::string bias_dim2 = EmitCMSISNNDims(stream, "bias2", bias_dims2);
    std::string output_dim2 = EmitCMSISNNDims(stream, "output2", output_dims2);

    std::string conv_params3 = FITIConvParams(stream, conv2d_params3, "3");
    std::string quant_params3 = FITIQuantParams(stream, multiplier3, shift3, "3");
    std::string input_dim3 = EmitCMSISNNDims(stream, "input3", input_dims3);
    std::string filter_dim3 = EmitCMSISNNDims(stream, "filter3", filter_dims3);
    std::string bias_dim3 = EmitCMSISNNDims(stream, "bias3", bias_dims3);
    std::string output_dim3 = EmitCMSISNNDims(stream, "output3", output_dims3);

    // Emit CMSIS-NN API
    PrintIndent();
    stream << "arm_status status = ";
    //stream << cmsis_func_name << "(";
    stream << "fiti_convolve_wrapper_s83(";
    stream << input_data1 << ", ";
    stream << "&" << conv_params1 << ", ";
    stream << "&" << quant_params1 << ", ";
    stream << "&" << input_dim1 << ", ";
    stream << "&" << filter_dim1 << ", " << filter_data1 << ", ";
    stream << "&" << bias_dim1 << ", " << bias_data1 << ", ";
    stream << "&" << output_dim1 << ",\n";
    stream << "&" << conv_params2 << ", ";
    stream << "&" << quant_params2 << ", ";
    stream << "&" << input_dim2 << ", ";
    stream << "&" << filter_dim2 << ", " << filter_data2 << ", ";
    stream << "&" << bias_dim2 << ", " << bias_data2 << ", ";
    stream << "&" << output_dim2 << ",\n ";
    stream << "&" << conv_params3 << ", ";
    stream << "&" << quant_params3 << ", ";
    stream << "&" << input_dim3 << ", ";
    stream << "&" << filter_dim3 << ", " << filter_data3 << ", ";
    stream << "&" << bias_dim3 << ", " << bias_data3 << ", ";
    stream << "&" << output_dim3 << ", " << output_data3 << ");\n";
    EmitErrorCheck();
  }

  void FITI_reduce_transaction2(const CallNode* op) 
  {
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;
    // extract buffer names from call_extern
    int arg_id = 0;
    
    std::string input_data1 = VarNameFromArg(op, ++arg_id);
    std::string filter_data1 = VarNameFromArg(op, ++arg_id);
    std::string multiplier1 = VarNameFromArg(op, ++arg_id);
    std::string bias_data1 = VarNameFromArg(op, ++arg_id);
    std::string shift1 = VarNameFromArg(op, ++arg_id);
    //std::string output_data1 = VarNameFromArg(op, ++arg_id);
    //std::string input_data2 = VarNameFromArg(op, ++arg_id);
    std::string filter_data2 = VarNameFromArg(op, ++arg_id);
    std::string multiplier2 = VarNameFromArg(op, ++arg_id);
    std::string bias_data2 = VarNameFromArg(op, ++arg_id);
    std::string shift2 = VarNameFromArg(op, ++arg_id);
    std::string output_data2 = VarNameFromArg(op, ++arg_id);

    Conv2DParams conv2d_params1 = extract_conv2d_params(op, ++arg_id);
    Conv2DParams conv2d_params2 = extract_conv2d_params(op, arg_id+11);
    CMSISNNDims input_dims1 = extract_buffer_dims(op, arg_id+22);
    CMSISNNDims filter_dims1 = extract_buffer_dims(op, arg_id+26);
    CMSISNNDims bias_dims1 = extract_buffer_dims(op, arg_id+30);
    CMSISNNDims output_dims1 = extract_buffer_dims(op, arg_id+34);
    CMSISNNDims input_dims2 = extract_buffer_dims(op, arg_id+38);
    CMSISNNDims filter_dims2 = extract_buffer_dims(op, arg_id+42);
    CMSISNNDims bias_dims2 = extract_buffer_dims(op, arg_id+46);
    CMSISNNDims output_dims2 = extract_buffer_dims(op, arg_id+50);

    if(filter_dims1.c != input_dims1.c)
    {
      filter_dims1.h = std::sqrt(filter_dims1.c / input_dims1.c);
      filter_dims1.w = std::sqrt(filter_dims1.c / input_dims1.c);
      filter_dims1.c = input_dims1.c;
    }

    if(filter_dims2.c != input_dims2.c)
    {
      filter_dims2.h = std::sqrt(filter_dims2.c / input_dims2.c);
      filter_dims2.w = std::sqrt(filter_dims2.c / input_dims2.c);
      filter_dims2.c = input_dims2.c;
    }

    // Emit CMSIS-NN API arguments
    std::string conv_params1 = FITIConvParams(stream, conv2d_params1, "1");
    std::string quant_params1 = FITIQuantParams(stream, multiplier1, shift1, "1");
    std::string input_dim1 = EmitCMSISNNDims(stream, "input1", input_dims1);
    std::string filter_dim1 = EmitCMSISNNDims(stream, "filter1", filter_dims1);
    std::string bias_dim1 = EmitCMSISNNDims(stream, "bias1", bias_dims1);
    std::string output_dim1 = EmitCMSISNNDims(stream, "output1", output_dims1);

    std::string conv_params2 = FITIConvParams(stream, conv2d_params2, "2");
    std::string quant_params2 = FITIQuantParams(stream, multiplier2, shift2, "2");
    std::string input_dim2 = EmitCMSISNNDims(stream, "input2", input_dims2);
    std::string filter_dim2 = EmitCMSISNNDims(stream, "filter2", filter_dims2);
    std::string bias_dim2 = EmitCMSISNNDims(stream, "bias2", bias_dims2);
    std::string output_dim2 = EmitCMSISNNDims(stream, "output2", output_dims2);

    // Emit CMSIS-NN API
    PrintIndent();
    stream << "arm_status status = ";
    //stream << cmsis_func_name << "(";
    stream << "fiti_convolve_wrapper_s82(";
    stream << input_data1 << ", ";
    stream << "&" << conv_params1 << ", ";
    stream << "&" << quant_params1 << ", ";
    stream << "&" << input_dim1 << ", ";
    stream << "&" << filter_dim1 << ", " << filter_data1 << ", ";
    stream << "&" << bias_dim1 << ", " << bias_data1 << ", ";
    stream << "&" << output_dim1 << ",\n";
    stream << "&" << conv_params2 << ", ";
    stream << "&" << quant_params2 << ", ";
    stream << "&" << input_dim2 << ", ";
    stream << "&" << filter_dim2 << ", " << filter_data2 << ", ";
    stream << "&" << bias_dim2 << ", " << bias_data2 << ", ";
    stream << "&" << output_dim2 << ", " << output_data2 << ");\n";
    EmitErrorCheck();
  }

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern comprising convolution */
  void EmitConv2D(const CallNode* op) {
    // Position of various arguments relative to buffers in the call_extern
    enum CallExternArgPos {
      CONTEXT_BUFFER_POS = 1,
      CONV2D_PARAMS_POS = 3,
      INPUT_DIM_POS = 14,
      FILTER_DIM_POS = 18,
      BIAS_DIM_POS = 22,
      OUTPUT_DIM_POS = 26,
      MAX_NUM_ARGS = 36
    };

    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;

    // extract buffer names from call_extern
    int arg_id = 0;
    std::string input_data = VarNameFromArg(op, ++arg_id);
    std::string filter_data = VarNameFromArg(op, ++arg_id);
    std::string multiplier = VarNameFromArg(op, ++arg_id);
    std::string bias_data("NULL");
    if (op->args.size() == CallExternArgPos::MAX_NUM_ARGS) {
      bias_data = VarNameFromArg(op, ++arg_id);
    }
    std::string shift = VarNameFromArg(op, ++arg_id);
    std::string output_data = VarNameFromArg(op, ++arg_id);

    // extract CMSIS-NN API parameters
    int context_buffer_pos = arg_id + CallExternArgPos::CONTEXT_BUFFER_POS;
    int conv2d_params_pos = arg_id + CallExternArgPos::CONV2D_PARAMS_POS;
    int input_dim_pos = arg_id + CallExternArgPos::INPUT_DIM_POS;
    int filter_dim_pos = arg_id + CallExternArgPos::FILTER_DIM_POS;
    int bias_dim_pos = arg_id + CallExternArgPos::BIAS_DIM_POS;
    int output_dim_pos = arg_id + CallExternArgPos::OUTPUT_DIM_POS;

    CMSISNNContextBuffer context_buffer = extract_context_buffer_info(op, context_buffer_pos);
    Conv2DParams conv2d_params = extract_conv2d_params(op, conv2d_params_pos);
    CMSISNNDims input_dims = extract_buffer_dims(op, input_dim_pos);
    CMSISNNDims filter_dims = extract_buffer_dims(op, filter_dim_pos);
    CMSISNNDims bias_dims = extract_buffer_dims(op, bias_dim_pos);
    CMSISNNDims output_dims = extract_buffer_dims(op, output_dim_pos);

    if(filter_dims.c != input_dims.c)
    {
      filter_dims.h = std::sqrt(filter_dims.c / input_dims.c);
      filter_dims.w = std::sqrt(filter_dims.c / input_dims.c);
      filter_dims.c = input_dims.c;
    }

    // Emit CMSIS-NN API arguments
    std::string context = EmitCMSISNNContext(stream, context_buffer);
    std::string conv_params = EmitCMSISNNConvParams(stream, conv2d_params);
    std::string quant_params = EmitCMSISNNPerChannelQuantParams(stream, multiplier, shift);
    std::string input_dim = EmitCMSISNNDims(stream, "input", input_dims);
    std::string filter_dim = EmitCMSISNNDims(stream, "filter", filter_dims);
    std::string bias_dim = EmitCMSISNNDims(stream, "bias", bias_dims);
    std::string output_dim = EmitCMSISNNDims(stream, "output", output_dims);

    // Emit CMSIS-NN API
    PrintIndent();
    stream << "arm_cmsis_nn_status status = ";
    stream << cmsis_func_name << "(";
    stream << "&" << context << ", ";
    stream << "&" << conv_params << ", ";
    stream << "&" << quant_params << ", ";
    stream << "&" << input_dim << ", " << input_data << ", ";
    stream << "&" << filter_dim << ", " << filter_data << ", ";
    stream << "&" << bias_dim << ", " << bias_data << ", ";
    stream << "&" << output_dim << ", " << output_data << ");\n";
    EmitErrorCheck();
  }

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern comprising fully connected */
  void EmitFullyConnected(const CallNode* op) {
    // Position of various arguments relative to buffers in the call_extern
    enum CallExternArgPos {
      CONTEXT_BUFFER_POS = 1,
      FC_PARAMS_POS = 3,
      INPUT_DIM_POS = 10,
      FILTER_DIM_POS = 14,
      BIAS_DIM_POS = 18,
      OUTPUT_DIM_POS = 22,
      MAX_NUM_ARGS = 30
    };

    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;

    // extract buffer names from call_extern
    int arg_id = 0;
    std::string input_data = VarNameFromArg(op, ++arg_id);
    std::string filter_data = VarNameFromArg(op, ++arg_id);
    std::string bias_data("NULL");
    if (op->args.size() == CallExternArgPos::MAX_NUM_ARGS) {
      bias_data = VarNameFromArg(op, ++arg_id);
    }
    std::string output_data = VarNameFromArg(op, ++arg_id);

    // extract CMSIS-NN API parameters
    int context_buffer_pos = arg_id + CallExternArgPos::CONTEXT_BUFFER_POS;
    int fc_params_pos = arg_id + CallExternArgPos::FC_PARAMS_POS;
    int input_dim_pos = arg_id + CallExternArgPos::INPUT_DIM_POS;
    int filter_dim_pos = arg_id + CallExternArgPos::FILTER_DIM_POS;
    int bias_dim_pos = arg_id + CallExternArgPos::BIAS_DIM_POS;
    int output_dim_pos = arg_id + CallExternArgPos::OUTPUT_DIM_POS;

    CMSISNNContextBuffer context_buffer = extract_context_buffer_info(op, context_buffer_pos);
    FCParams fc_params = extract_fc_params(op, fc_params_pos);
    CMSISNNDims input_dims = extract_buffer_dims(op, input_dim_pos);
    CMSISNNDims filter_dims = extract_buffer_dims(op, filter_dim_pos);
    CMSISNNDims bias_dims = extract_buffer_dims(op, bias_dim_pos);
    CMSISNNDims output_dims = extract_buffer_dims(op, output_dim_pos);

    // Emit CMSIS-NN API arguments
    std::string context = EmitCMSISNNContext(stream, context_buffer);
    std::string cmsisnn_fc_params = EmitCMSISNNFCParams(stream, fc_params);
    std::string quant_params =
        EmitCMSISNNPerTensorQuantParams(stream, fc_params.multiplier, fc_params.shift);
    std::string input_dim = EmitCMSISNNDims(stream, "input", input_dims);
    std::string filter_dim = EmitCMSISNNDims(stream, "filter", filter_dims);
    std::string bias_dim = EmitCMSISNNDims(stream, "bias", bias_dims);
    std::string output_dim = EmitCMSISNNDims(stream, "output", output_dims);

    PrintIndent();
    stream << "arm_cmsis_nn_status status = ";
    stream << cmsis_func_name << "(";
    stream << "&" << context << ", ";
    stream << "&" << cmsisnn_fc_params << ", ";
    stream << "&" << quant_params << ", ";
    stream << "&" << input_dim << ", " << input_data << ", ";
    stream << "&" << filter_dim << ", " << filter_data << ", ";
    stream << "&" << bias_dim << ", " << bias_data << ", ";
    stream << "&" << output_dim << ", " << output_data << ");\n";
    EmitErrorCheck();
  }

  /*!  * \brief Emits CMSIS-NN APIs for every call_extern comprising pooling ops */
  void EmitPool2D(const CallNode* op) {
    // Position of various arguments relative to buffers in the call_extern
    enum CallExternArgPos {
      CONTEXT_BUFFER_POS = 1,
      POOL_PARAMS_POS = 3,
      INPUT_DIM_POS = 9,
      FILTER_DIM_POS = 13,
      OUTPUT_DIM_POS = 17,
      MAX_NUM_ARGS = 23
    };
    std::string cmsis_func_name = op->args[0].as<StringImmNode>()->value;

    // extract buffer names from call_extern
    int arg_id = 0;
    std::string input_data = VarNameFromArg(op, ++arg_id);
    std::string output_data = VarNameFromArg(op, ++arg_id);

    // extract CMSIS-NN API parameters
    int context_buffer_pos = arg_id + CallExternArgPos::CONTEXT_BUFFER_POS;
    int pool_params_pos = arg_id + CallExternArgPos::POOL_PARAMS_POS;
    int input_dim_pos = arg_id + CallExternArgPos::INPUT_DIM_POS;
    int filter_dim_pos = arg_id + CallExternArgPos::FILTER_DIM_POS;
    int output_dim_pos = arg_id + CallExternArgPos::OUTPUT_DIM_POS;

    CMSISNNContextBuffer context_buffer = extract_context_buffer_info(op, context_buffer_pos);
    PoolParams pool_params = extract_pool_params(op, pool_params_pos);
    CMSISNNDims input_dims = extract_buffer_dims(op, input_dim_pos);
    CMSISNNDims filter_dims = extract_buffer_dims(op, filter_dim_pos);
    CMSISNNDims output_dims = extract_buffer_dims(op, output_dim_pos);

    std::string context = EmitCMSISNNContext(stream, context_buffer);
    std::string cmsisnn_pool_params = EmitCMSISNNPoolParams(stream, pool_params);
    std::string input_dim = EmitCMSISNNDims(stream, "input", input_dims);
    std::string filter_dim = EmitCMSISNNDims(stream, "filter", filter_dims);
    std::string output_dim = EmitCMSISNNDims(stream, "output", output_dims);

    PrintIndent();
    stream << "arm_cmsis_nn_status status = ";
    stream << cmsis_func_name << "(";
    stream << "&" << context << ", ";
    stream << "&" << cmsisnn_pool_params << ", ";
    stream << "&" << input_dim << ", " << input_data << ", ";
    stream << "&" << filter_dim << ", ";
    stream << "&" << output_dim << ", " << output_data << ");\n";
    EmitErrorCheck();
  }

  void EmitErrorCheck() {
    auto emit_error = [&](std::string error) {
      if (this->debug_last_error) {
        stream << "TVMAPISetLastError(\"" << error << "\"); ";
      }
    };

    PrintIndent();
    stream << "switch (!status) {\n";
    PrintIndent();
    stream << "case ARM_CMSIS_NN_SUCCESS: break;\n";
    PrintIndent();
    stream << "case ARM_CMSIS_NN_ARG_ERROR: ";
    emit_error("ARM_CMSIS_NN_ARG_ERROR");
    stream << "return -1;\n";
    PrintIndent();
    stream << "case ARM_CMSIS_NN_NO_IMPL_ERROR: ";
    emit_error("ARM_CMSIS_NN_NO_IMPL_ERROR");
    stream << "return -1;\n";
    PrintIndent();
    stream << "}\n";
  }
};

static CMSISNNCompilerConfig GetCompilerAttrs() {
  auto ctx = tvm::tir::transform::PassContext::Current();
  Optional<CMSISNNCompilerConfig> cfg =
      ctx->GetConfig<CMSISNNCompilerConfig>("relay.ext.cmsisnn.options");
  if (!cfg.defined()) {
    return AttrsWithDefaultValues<CMSISNNCompilerConfig>();
  }
  return cfg.value();
}

runtime::Module TIRToRuntime(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = false;
  bool debug_last_error = GetCompilerAttrs()->debug_last_error;
  CodeGenCMSISNN codegen;
  Array<String> function_names;
  codegen.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), debug_last_error);
  std::vector<std::pair<tvm::GlobalVar, tvm::BaseFunc>> funcs;
  for (auto kv : mod->functions) {
    funcs.push_back(kv);
  }

  std::sort(funcs.begin(), funcs.end(),
            [](std::pair<tvm::GlobalVar, tvm::BaseFunc> kv_a,
               std::pair<tvm::GlobalVar, tvm::BaseFunc> kv_b) {
              std::string name_hint_a = kv_a.first->name_hint;
              std::string name_hint_b = kv_b.first->name_hint;
              size_t name_a_length = name_hint_a.length();
              size_t name_b_length = name_hint_b.length();
              if (name_a_length < name_b_length) return true;
              if (name_a_length > name_b_length) return false;
              return name_hint_a < name_hint_b;
            });

  for (auto kv : funcs) {
    auto prim_func = Downcast<PrimFunc>(kv.second);
    auto global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    function_names.push_back(global_symbol.value());
    codegen.AddFunction(prim_func);
  }
  std::string code = codegen.Finish();
  return codegen::CSourceModuleCreate(code, "c", function_names);
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
