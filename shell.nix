
let
  pkgs = import <nixpkgs> {};

  # bring in yellowbrick from pypi, building it with a recursive list
    yellowbrick = pkgs.python37.pkgs.buildPythonPackage rec {
      pname = "yellowbrick";
      version = "1.0.1" ;

      src = pkgs.python37.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "1q659ayr657p786gwrh11lw56jw5bdpnl6hp11qlckvh15haywvk";
      };

# no tests because this is a simple example
      doCheck = false;
# dependencies for yellowbrick
      buildInputs = with pkgs.python37Packages; [
      pytest 
      pytestrunner 
      pytest-flakes  
      numpy 
      matplotlib 
      scipy 
      scikitlearn
    ];
  };

in
  pkgs.mkShell {
    name = "ML";
    buildInputs = with pkgs; [
      python37
      python37Packages.numpy
      python37Packages.scikitlearn
      zip
      python37Packages.scipy
      python37Packages.matplotlib
      python37Packages.seaborn
      yellowbrick
    ];
   shellHook = ''
      '';

  }
