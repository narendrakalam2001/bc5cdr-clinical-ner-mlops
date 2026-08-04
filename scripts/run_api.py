# ============================================================
# RUN API — BC5CDR Clinical NER System
# ============================================================
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="BC5CDR Clinical NER API")
    parser.add_argument("--host",   type=str,  default="0.0.0.0")
    parser.add_argument("--port",   type=int,  default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn
    print(f"\n🧬 BC5CDR Clinical NER API starting on http://{args.host}:{args.port}")
    print(f"   Docs  : http://localhost:{args.port}/docs")
    print(f"   Health: http://localhost:{args.port}/health\n")

    uvicorn.run("serving.ner_api:app",
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
