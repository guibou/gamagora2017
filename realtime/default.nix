{ system ? builtins.currentSystem }:

let
  pkgs = import <nixpkgs> { inherit system; };
in
rec {
  rt = (pkgs.overrideCC pkgs.stdenv pkgs.gcc7).mkDerivation {
    name = "rt";

    src = ./.;

    hardeningDisable = [ "all" ];

    buildInputs = [ pkgs.glfw pkgs.glm ];
  };
}
