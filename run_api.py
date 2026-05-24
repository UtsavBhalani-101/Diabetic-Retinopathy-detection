# run_api.py
# ============================================================
# Entry point to launch the APTOS DR Inference API server.
#
# Usage:
#   python run_api.py                          # localhost:8000
#   python run_api.py --host 0.0.0.0           # expose to network
#   python run_api.py --port 9000              # custom port
#   python run_api.py --reload                 # hot-reload (dev only)
#
# Once running:
#   Interactive docs  →  http://localhost:8000/docs
#   Health check      →  http://localhost:8000/health
#   Model info        →  http://localhost:8000/model/info
#   Predict           →  POST http://localhost:8000/predict
# ============================================================

import argparse
import logging

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="APTOS Diabetic Retinopathy Inference API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        type=str,
        help="Host address to bind the server to. "
             "Use '0.0.0.0' to expose on the local network.",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="Port number to listen on.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload on source file changes. "
             "For development only — do NOT use in production.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Uvicorn log level.",
    )

    args = parser.parse_args()

    # Configure Python root logger so startup messages appear before uvicorn takes over
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    sep = "-" * 45
    print(f"\n  APTOS DR Inference API")
    print(f"  {sep}")
    print(f"  Host      : {args.host}")
    print(f"  Port      : {args.port}")
    print(f"  Docs      : http://{args.host}:{args.port}/docs")
    print(f"  Health    : http://{args.host}:{args.port}/health")
    print(f"  Model info: http://{args.host}:{args.port}/model/info")
    print(f"  Hot-reload: {args.reload}")
    print(f"  {sep}\n")

    uvicorn.run(
        "api.app:app",
        host      = args.host,
        port      = args.port,
        reload    = args.reload,
        log_level = args.log_level,
    )


if __name__ == "__main__":
    main()
