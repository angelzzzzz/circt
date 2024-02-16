//===- MIRExpressions.td - Moore MIR expression ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR expressions.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/Moore/MooreTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Base class for binary operators.
class BinOp<string mnemonic, list<Trait> traits = []> :
      MIROp<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, UnitAttr:$twoState);
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs (`bin` $twoState^)? attr-dict `:` functional-type($args, $results)";
}

//===---------------------------------------------------------------------===//
// Equality operations
//===---------------------------------------------------------------------===//

class EqualOp<string name> : MIROp<name, [
  Pure,
  Commutative
]> {
  let arguments = (ins SimpleBitVectorType:$lhs,
                       SimpleBitVectorType:$rhs,
                       UnitAttr:$mode);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    ( `case` $mode^ )? $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def EqualityOp : EqualOp<"eq">{
  let summary = "A logical or case equality expression";
  let description = [{
    This operation represents the SystemVerilog logical and case
    equality expressions `==` and `===`.
    See IEEE1800-2017 11.4.5 "Equality operators".

    a == ba equal to b, result can be unknown.
    a === ba equal to b, including x and z.

    The operators compare operands bit for bit. As with the relational
    operators, the result shall be 0 if comparison fails and 1 if it succeeds.
  }];
}

def InEqualityOp : EqualOp<"ne">{
  let summary = "A logical or case inequality expression";
  let description = [{
    This operation represents the SystemVerilog logical and case
    inequality expressions `!=` and `!==`.
    See IEEE1800-2017 11.4.5 "Equality operators".

    a != ba not equal to b, result can be unknown.
    a !== ba not equal to b, including x and z.
  }];
}
