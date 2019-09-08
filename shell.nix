
let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    name = "ML";
    buildInputs = with pkgs; [
      python37
      python37Packages.numpy
      python37Packages.scikitlearn
      python37Packages.scipy
      python37Packages.pip
      python37Packages.virtualenv
    ];
   shellHook = ''
      '';

  }
