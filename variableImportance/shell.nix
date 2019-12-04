let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    name = "varimp";
    buildInputs = with pkgs; [
      R
      rPackages.rmarkdown
      rPackages.magick
      rPackages.revealjs
      rPackages.knitr
      rPackages.tidyverse
      rPackages.ggthemes
      rPackages.plotly
      rPackages.highcharter
      rPackages.rmdformats
      rPackages.iml
      rPackages.reticulate
      rPackages.kknn
      rPackages.ranger
      rPackages.gbm
    ];
   shellHook = ''
      '';

  }
