// RUN: circt-translate --import-verilog %s | FileCheck %s


// CHECK-LABEL: moore.module @Variables
module Variables();
  // CHECK: %var1 = moore.variable : !moore.int
  // CHECK: %var2 = moore.variable %var1 : !moore.int
  int var1;
  int var2 = var1;
endmodule


// CHECK-LABEL: moore.module @Procedures
module Procedures();
  // CHECK: moore.procedure initial {
  initial;
  // CHECK: moore.procedure final {
  final begin end;
  // CHECK: moore.procedure always {
  always begin end;
  // CHECK: moore.procedure always_comb {
  always_comb begin end;
  // CHECK: moore.procedure always_latch {
  always_latch begin end;
  // CHECK: moore.procedure always_ff {
  always_ff @* begin end;
endmodule


// CHECK-LABEL: moore.module @Expressions {
module Expressions();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  // CHECK: %c = moore.variable : !moore.int
  int a, b, c;
  bit [1:0][3:0] v;
  integer d, e, f;
  bit x;
  logic y;

  initial begin
    // CHECK: moore.constant 42 : !moore.int
    c = 42;

    // Unary operators

    // CHECK: moore.mir.bpassign %c, %a : !moore.int
    c = a;
    // CHECK: moore.neg %a : !moore.int
    c = -a;
    // CHECK: [[TMP1:%.+]] = moore.conversion %v : !moore.packed<range<range<bit, 3:0>, 1:0>> -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP2:%.+]] = moore.neg [[TMP1]] : !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.packed<range<bit, 31:0>> -> !moore.int
    c = -v;
    // CHECK: moore.not %a : !moore.int
    c = ~a;

    // CHECK: moore.reduce_and %a : !moore.int -> !moore.bit
    x = &a;
    // CHECK: moore.reduce_and %d : !moore.integer -> !moore.logic
    y = &d;
    // CHECK: moore.reduce_or %a : !moore.int -> !moore.bit
    x = |a;
    // CHECK: moore.reduce_xor %a : !moore.int -> !moore.bit
    x = ^a;
    // CHECK: [[TMP:%.+]] = moore.reduce_and %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~&a;
    // CHECK: [[TMP:%.+]] = moore.reduce_or %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~|a;
    // CHECK: [[TMP:%.+]] = moore.reduce_xor %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~^a;
    // CHECK: [[TMP:%.+]] = moore.reduce_xor %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ^~a;
    // CHECK: [[TMP:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = !a;

    // CHECK: moore.mir.bpassign %c, %a : !moore.int
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[TMP2:%.+]] = moore.add %a, [[TMP1]] : !moore.int
    // CHECK: moore.mir.bpassign %a, [[TMP2]] : !moore.int
    c = a++;
    // CHECK: moore.mir.bpassign %c, %a : !moore.int
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[TMP2:%.+]] = moore.sub %a, [[TMP1]] : !moore.int
    // CHECK: moore.mir.bpassign %a, [[TMP2]] : !moore.int
    c = a--;
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[TMP2:%.+]] = moore.add %a, [[TMP1]] : !moore.int
    // CHECK: moore.mir.bpassign %a, [[TMP2]] : !moore.int
    // CHECK: moore.mir.bpassign %c, [[TMP2]] : !moore.int
    c = ++a;
    // CHECK: [[TMP1:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[TMP2:%.+]] = moore.sub %a, [[TMP1]] : !moore.int
    // CHECK: moore.mir.bpassign %a, [[TMP2]] : !moore.int
    // CHECK: moore.mir.bpassign %c, [[TMP2]] : !moore.int
    c = --a;

    // Binary operators

    // CHECK: moore.add %a, %b : !moore.int
    c = a + b;
    // CHECK: [[TMP1:%.+]] = moore.conversion %a : !moore.int -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP2:%.+]] = moore.conversion %v : !moore.packed<range<range<bit, 3:0>, 1:0>> -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP3:%.+]] = moore.add [[TMP1]], [[TMP2]] : !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.packed<range<bit, 31:0>> -> !moore.int
    c = a + v;
    // CHECK: moore.sub %a, %b : !moore.int
    c = a - b;
    // CHECK: moore.mul %a, %b : !moore.int
    c = a * b;
    // CHECK: moore.div %d, %e : !moore.integer
    f = d / e;
    // CHECK: moore.mod %d, %e : !moore.integer
    f = d % e;

    // CHECK: moore.and %a, %b : !moore.int
    c = a & b;
    // CHECK: moore.or %a, %b : !moore.int
    c = a | b;
    // CHECK: moore.xor %a, %b : !moore.int
    c = a ^ b;
    // CHECK: [[TMP:%.+]] = moore.xor %a, %b : !moore.int
    // CHECK: moore.not [[TMP]] : !moore.int
    c = a ~^ b;
    // CHECK: [[TMP:%.+]] = moore.xor %a, %b : !moore.int
    // CHECK: moore.not [[TMP]] : !moore.int
    c = a ^~ b;

    // CHECK: moore.eq %a, %b : !moore.int -> !moore.bit
    x = a == b;
    // CHECK: moore.eq %d, %e : !moore.integer -> !moore.logic
    y = d == e;
    // CHECK: moore.ne %a, %b : !moore.int -> !moore.bit
    x = a != b ;
    // CHECK: moore.case_eq %a, %b : !moore.int
    x = a === b;
    // CHECK: moore.case_ne %a, %b : !moore.int
    x = a !== b;
    // CHECK: moore.wildcard_eq %a, %b : !moore.int -> !moore.bit
    x = a ==? b;
    // CHECK: [[TMP:%.+]] = moore.conversion %a : !moore.int -> !moore.integer
    // CHECK: moore.wildcard_eq [[TMP]], %d : !moore.integer -> !moore.logic
    y = a ==? d;
    // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.integer
    // CHECK: moore.wildcard_eq %d, [[TMP]] : !moore.integer -> !moore.logic
    y = d ==? b;
    // CHECK: moore.wildcard_eq %d, %e : !moore.integer -> !moore.logic
    y = d ==? e;
    // CHECK: moore.wildcard_ne %a, %b : !moore.int -> !moore.bit
    x = a !=? b;

    // CHECK: moore.ge %a, %b : !moore.int -> !moore.bit
    c = a >= b;
    // CHECK: moore.gt %a, %b : !moore.int -> !moore.bit
    c = a > b;
    // CHECK: moore.le %a, %b : !moore.int -> !moore.bit
    c = a <= b;
    // CHECK: moore.lt %a, %b : !moore.int -> !moore.bit
    c = a < b;

    // CHECK: moore.logical_and %a, %b : !moore.int, !moore.int -> !moore.bit
    c = a && b;
    // CHECK: moore.logical_equiv %a, %b : !moore.int, !moore.int -> !moore.bit
    c = a <-> b;
    // CHECK: moore.logical_impl %a, %b : !moore.int, !moore.int -> !moore.bit
    c = a -> b;
    // CHECK: moore.logical_or %a, %b : !moore.int, !moore.int -> !moore.bit
    c = a || b;

    // CHECK: moore.mir.shl %a, %b : !moore.int, !moore.int
    c = a << b;
    // CHECK: moore.mir.shr %a, %b : !moore.int, !moore.int
    c = a >> b;
    // CHECK: moore.mir.shl arithmetic %a, %b : !moore.int, !moore.int
    c = a <<< b;
    // CHECK: moore.mir.shr arithmetic %a, %b : !moore.int, !moore.int
    c = a >>> b;

    // CHECK: [[TMP1:%.+]] = moore.gt %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.bit -> i1
    // CHECK: [[TMP3:%.+]] = scf.if [[TMP2]] -> (!moore.int) {
    // CHECK:   scf.yield %a : !moore.int
    // CHECK: } else {
    // CHECK:   scf.yield %b : !moore.int
    // CHECK: }
    // CHECK: moore.mir.bpassign %c, [[TMP3]] : !moore.int
    c = a > b ? a : b;

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.bit -> !moore.packed<range<logic, 31:0>>
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.packed<range<logic, 31:0>> -> !moore.int
    // CHECK: moore.mir.bpassign %c, [[TMP3]] : !moore.int
    c = a inside {a};

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP3:%.+]] = moore.logical_or [[TMP1]], [[TMP2]] : !moore.bit, !moore.bit -> !moore.bit
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.bit -> !moore.packed<range<logic, 31:0>>
    // CHECK: [[TMP5:%.+]] = moore.conversion [[TMP4]] : !moore.packed<range<logic, 31:0>> -> !moore.int
    // CHECK: moore.mir.bpassign %c, [[TMP5]] : !moore.int
    c = a inside {a, b};

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP3:%.+]] = moore.logical_or [[TMP1]], [[TMP2]] : !moore.bit, !moore.bit -> !moore.bit
    // CHECK: [[TMP4:%.+]] = moore.eq %a, %a : !moore.int -> !moore.bit
    // CHECK: [[TMP5:%.+]] = moore.logical_or [[TMP3]], [[TMP4]] : !moore.bit, !moore.bit -> !moore.bit
    // CHECK: [[TMP6:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP7:%.+]] = moore.logical_or [[TMP5]], [[TMP6]] : !moore.bit, !moore.bit -> !moore.bit
    // CHECK: [[TMP8:%.+]] = moore.conversion [[TMP7]] : !moore.bit -> !moore.packed<range<logic, 31:0>>
    // CHECK: [[TMP9:%.+]] = moore.conversion [[TMP8]] : !moore.packed<range<logic, 31:0>> -> !moore.int
    // CHECK: moore.mir.bpassign %c, [[TMP9]] : !moore.int
    c = a inside {a, b, a, b};

  end
endmodule


// CHECK-LABEL: moore.module @Conversion {
module Conversion();
  // Implicit conversion.
  // CHECK: %a = moore.variable
  // CHECK: [[TMP:%.+]] = moore.conversion %a : !moore.shortint -> !moore.int
  // CHECK: %b = moore.variable [[TMP]]
  shortint a;
  int b = a;

  // Explicit conversion.
  // CHECK: [[TMP1:%.+]] = moore.conversion %a : !moore.shortint -> !moore.byte
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.byte -> !moore.int
  // CHECK: %c = moore.variable [[TMP2]]
  int c = byte'(a);

  // Sign conversion.
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit<signed>, 31:0>>
  // CHECK: %d1 = moore.variable [[TMP]]
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit, 31:0>>
  // CHECK: %d2 = moore.variable [[TMP]]
  bit signed [31:0] d1 = signed'(b);
  bit [31:0] d2 = unsigned'(b);

  // Width conversion.
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit<signed>, 18:0>>
  // CHECK: %e = moore.variable [[TMP]]
  bit signed [18:0] e = 19'(b);
endmodule


// CHECK-LABEL: moore.module @Assignments {
module Assignments();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  int a, b;

  initial begin
    // CHECK: moore.mir.bpassign %a, %b : !moore.int
    a = b;
    // CHECK: moore.mir.passign %a, %b : !moore.int
    a <= b;
    // CHECK: moore.mir.pcassign %a, %b : !moore.int
    assign a = b;
  end
endmodule


// CHECK-LABEL: moore.module @Statements {
module Statements();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  int a, b;

  initial begin

    // CHECK: [[TMP1:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP2:%.+]]
    if (a)
      ;

    // CHECK: [[TMP1:%.+]] = moore.eq %a, %b : !moore.int -> !moore.bit
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP2]] {
    // CHECK: }
    // CHECK: [[TMP3:%.+]] = moore.not [[TMP1]] : !moore.bit
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.bit -> i1
    // CHECK: scf.if [[TMP4]] {
    // CHECK: }
    case (a)
      b: ;
      default ;
    endcase
    
  end
endmodule


// CHECK-LABEL: moore.module @Generates {
module Generates();
  // CHECK: %a = moore.named_constant parameter 2 : !moore.packed<range<logic<signed>, 31:0>>
  // CHECK: %b = moore.named_constant localparam 3 : !moore.packed<range<logic<signed>, 31:0>>
  // CHECK：%sp = moore.named_constant specparam 4 : !moore.packed<range<logic<signed>, 31:0>>
  parameter a = 2;
  localparam b = 3;
  specparam sp = 4;
  genvar i;

  generate
    // CHECK: %i = moore.named_constant localparam 0 : !moore.integer
    // CHECK: [[TMP1:%.+]] = moore.conversion %i : !moore.integer -> !moore.int
    // CHECK: %c = moore.variable [[TMP1]] : !moore.int
    // CHECK: %i_0 = moore.named_constant name "i" localparam 1 : !moore.integer
    // CHECK: [[TMP2:%.+]] = moore.conversion %i_0 : !moore.integer -> !moore.int
    // CHECK: %c_1 = moore.variable name "c" [[TMP2]] : !moore.int
    for(i=0;i<a;i=i+1)begin
	    int c = i;
	  end

    // CHECK: [[TMP1:%.+]] = moore.constant 2 : !moore.int
    // CHECK: %d = moore.variable [[TMP1]] : !moore.int
	  if(a == 2)begin
      int d = 2;
	  end
  	else begin
      int d = 3;
	  end

    // CHECK: [[TMP1:%.+]] = moore.constant 2 : !moore.int
    // CHECK: %e = moore.variable [[TMP1]] : !moore.int
	  case (a)
      2:begin
        int e = 2;
      end
      default:begin
        int e = 3;
      end
    endcase
    
  endgenerate
endmodule