{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/a983cc62cc2345443620597697015c1c9c4e5b06";
    utils.url = "github:numtide/flake-utils/93a2b84fc4b70d9e089d029deacc3583435c2ed6";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      inherit (pkgs) bash buildEnv cudaPackages dockerTools linuxPackages mkShell poetry2nix python39 stdenv fetchurl;
      inherit (poetry2nix) mkPoetryApplication mkPoetryEnv;
      inherit (cudaPackages) cudatoolkit;
      inherit (linuxPackages) nvidia_x11;
      python = python39;
      overrides = pyfinal: pyprev: let
        inherit (pyprev) buildPythonPackage fetchPypi;
      in rec {
        torch =
          # Override the nixpkgs bin version instead of
          # poetry2nix version so that rpath is set correctly.
          pyprev.pytorch-bin.overridePythonAttrs (old: {
            inherit (old) pname version;
            src = fetchurl {
              url = "https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp39-cp39-linux_x86_64.whl";
              sha256 = "sha256-64HQZ7vP6ETJXF0n4myXqWqJNCfMRosiWerw7ZPaHH0=";
            };
          });
      };
      poetryArgs = {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };
      poetryEnv = mkPoetryEnv poetryArgs;
      buildInputs = with pkgs; [
        alejandra
        coreutils
        nodePackages.prettier
        poetry
        poetryEnv
      ];
    in rec {
      devShell = mkShell rec {
        inherit buildInputs;
        PYTHONFAULTHANDLER = 1;
        PYTHONBREAKPOINT = "ipdb.set_trace";
        LD_LIBRARY_PATH = "${nvidia_x11}/lib";
        shellHook = ''
          set -o allexport
          source .env
          set +o allexport
        '';
      };
      packages.default = dockerTools.buildImage {
        name = "ppo";
        tag = "latest";
        copyToRoot =
          buildEnv
          {
            name = "image-root";
            pathsToLink = ["/bin" "/ppo"];
            paths = buildInputs ++ [pkgs.git ./.];
          };
        config = {
          Env = with pkgs; [
            "PYTHONFAULTHANDLER=1"
            "PYTHONBREAKPOINT=ipdb.set_trace"
            "LD_LIBRARY_PATH=/usr/lib64/"
            "PATH=/bin:$PATH"
          ];
          Cmd = ["${poetryEnv.python}/bin/python"];
        };
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
