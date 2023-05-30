{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/a983cc62cc2345443620597697015c1c9c4e5b06";
    utils.url = "github:numtide/flake-utils/93a2b84fc4b70d9e089d029deacc3583435c2ed6";
    nixgl.url = "github:guibou/nixGL";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    nixgl,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
        overlays = [nixgl.overlay];
      };
      inherit (pkgs) poetry2nix;
      mujoco = pkgs.stdenv.mkDerivation rec {
        pname = "mujoco";
        version = "2.1.0";

        src = fetchTarball {
          url = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz";
          sha256 = "sha256:1lvppcdfca460sqnb0ryrach6lv1g9dwcjfim0dl4vmxg2ryaq7p";
        };
        installPhase = ''
          install -d $out/bin
          install -m 755 bin/* $out/bin/
          install -d $out/include
          install include/* $out/include/
        '';
        nativeBuildInputs = with pkgs;
          [glew110 autoPatchelfHook]
          ++ (with pkgs.xorg; [
            libXcursor
            libXinerama
            libXrandr
            libXxf86vm
          ]);
      };
      overrides = pyfinal: pyprev: rec {
        dm-alchemy = pyprev.dm-alchemy.overridePythonAttrs (old: {
          buildInputs = old.buildInputs ++ (with pyfinal; [setuptools grpcio-tools]);
        });
        grpcio-tools = pyprev.grpcio-tools.overridePythonAttrs (old: {
          buildInputs = [pyfinal.grpcio pkgs.gcc-unwrapped];
        });
        mujoco-py =
          (pyprev.mujoco-py.override {
            preferWheel = false;
          })
          .overridePythonAttrs (old: {
            env.NIX_CFLAGS_COMPILE = "-L${pkgs.mesa.osmesa}/lib";
            preBuild = with pkgs; ''
              export MUJOCO_PY_MUJOCO_PATH="${mujoco}"
              export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${mujoco}/bin:${pkgs.mesa.osmesa}/lib:${pkgs.libGL}/lib:${pkgs.gcc-unwrapped.lib}/lib
            '';
            buildInputs =
              old.buildInputs
              ++ [pyfinal.setuptools]
              ++ (with pkgs; [mesa libGL]);
            patches = [./mujoco-py.patch];
          });
        torch = pyprev.pytorch-bin.overridePythonAttrs (old: {
          src = pkgs.fetchurl {
            url = "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-linux_x86_64.whl";
            sha256 = "sha256-20V6gi1zYBO2/+UJBTABvJGL3Xj+aJZ7YF9TmEqa+sU=";
          };
        });
        torchrl = pyprev.buildPythonPackage {
          pname = "torchrl";
          version = "0.0.5";
          propagatedBuildInputs = with pyfinal; [
            tensordict
          ];
          nativeBuildInputs = with pyfinal; [
            setuptools
            cloudpickle
            packaging
            pkgs.which
          ];
          src = pkgs.fetchgit {
            url = "https://github.com/pytorch/rl";
            rev = "48eca9890e24a64b0f3ee133a74abafad7cd0768";
            sha256 = "sha256-ZLHqKRO7O4ZOSPqtwVmyFElpRnEbhX97Iy8PkGh4Fqg=";
            name = "torchrl";
          };
          doCheck = false;
          patches = [./torchrl.patch];
        };
      };
      poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
        python = pkgs.python39;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };

      myNixgl = pkgs.nixgl.override {
        nvidiaVersion = "510.108.03";
        nvidiaHash = "sha256-QQpRXnjfKcLLpKwLSXiJzg/xsEz8cR/4ieLfyA8NoNg=";
      };
    in {
      devShell = pkgs.mkShell {
        MUJOCO_PY_MUJOCO_PATH = "${mujoco}";
        MUJOCO_PY_FORCE_CPU = 1;
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python";
        LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${pkgs.cudaPackages.cudnn}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.mesa.osmesa}/lib:${pkgs.libGL}/lib:${pkgs.gcc-unwrapped.lib}/lib:${mujoco}/bin";
        buildInputs = with pkgs; [
          alejandra
          poetry
          poetryEnv
          myNixgl.nixGLNvidia
        ];
        PYTHONBREAKPOINT = "ipdb.set_trace";
        shellHook = ''
          set -o allexport
          source .env
          set +o allexport
        '';
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
