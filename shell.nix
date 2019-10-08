
let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    name = "ML";
    buildInputs = with pkgs; [
      python37
      python37Packages.numpy
      python37Packages.scikitlearn
      zip
      python37Packages.scipy
      python37Packages.pip
      python37Packages.virtualenv
      python37Packages.matplotlib
      python37Packages.seaborn
    ];
   shellHook = ''
    virtualenv ML
    source ML/bin/activate
    pip install -r requirements.txt
      '';

  }
