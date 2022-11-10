// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// expected-error @below {{expected ';'}}
module Foo 4;
endmodule

// expected-note @below {{expanded from macro 'FOO'}}
`define FOO input
// expected-note @below {{expanded from macro 'BAR'}}
`define BAR `FOO
// expected-error @below {{expected identifier}}
module Bar(`BAR);
endmodule

// -----

module Foo;
  // expected-warning @below {{unsupported construct ignored}}
  genvar a;
endmodule
