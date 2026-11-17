"""
Spy Game — servidor da plataforma.

Rode com:   python app.py
e abra      http://localhost:8000

Usa apenas a biblioteca padrão (http.server + json). O front-end (static/index.html)
chama a rota POST /api/executar, que roda o Bellman-Ford no Python e devolve
tudo em JSON.
"""

import json
import os
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from bellman_ford import INF, bellman_ford, reconstruir_rota

PORTA = int(os.environ.get("PORT", 8000))
PASTA_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _num(x):
    """JSON não aceita infinito: convertemos para string."""
    if x == INF:
        return "inf"
    if x == -INF:
        return "-inf"
    return x


def executar(payload):
    vertices = [str(v).strip() for v in payload.get("vertices", []) if str(v).strip()]
    if len(set(vertices)) != len(vertices):
        raise ValueError("Há esconderijos com nomes repetidos.")
    origem = payload.get("origem")
    if origem not in vertices:
        raise ValueError("Escolha um esconderijo de origem válido.")
    arestas = []
    for a in payload.get("arestas", []):
        u, v = a.get("origem"), a.get("destino")
        if u not in vertices or v not in vertices:
            raise ValueError(f"Rota {u} → {v} usa esconderijo inexistente.")
        try:
            w = float(a.get("peso"))
        except (TypeError, ValueError):
            raise ValueError(f"Peso inválido na rota {u} → {v}.")
        if w.is_integer():
            w = int(w)
        arestas.append((u, v, w))

    bf = bellman_ford(vertices, arestas, origem)

    esconderijos = []
    for v in vertices:
        d = bf["dist"][v]
        if d == -INF:
            status, rota = "armadilha", None
        elif d == INF:
            status, rota = "inalcancavel", None
        else:
            status, rota = "seguro", reconstruir_rota(bf["pred"], origem, v)
        esconderijos.append({"vertice": v, "custo": _num(d), "status": status, "rota": rota,
                             "pred": bf["pred"][v]})

    return {
        "origem": origem,
        "esconderijos": esconderijos,
        "tem_ciclo_negativo": bf["tem_ciclo_negativo"],
        "ciclos": bf["ciclos"],
        "afetados": bf["afetados"],
        "iteracoes": [
            {"iteracao": it["iteracao"],
             "dist": {k: _num(x) for k, x in it["dist"].items()},
             "relaxacoes": [{**r, "antes": _num(r["antes"]), "depois": _num(r["depois"])}
                            for r in it["relaxacoes"]]}
            for it in bf["iteracoes"]
        ],
        "verificacao": [{**r, "antes": _num(r["antes"]), "depois": _num(r["depois"])}
                        for r in bf["verificacao"]],
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PASTA_STATIC, **kwargs)

    def log_message(self, fmt, *args):  # terminal mais limpo
        pass

    def _json(self, status, corpo):
        dados = json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_POST(self):
        if self.path != "/api/executar":
            return self._json(404, {"erro": "Rota não encontrada."})
        try:
            tamanho = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(tamanho) or b"{}")
            return self._json(200, executar(payload))
        except ValueError as e:
            return self._json(400, {"erro": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"erro": f"Erro interno: {e}"})


def main():
    servidor = ThreadingHTTPServer(("127.0.0.1", PORTA), Handler)
    url = f"http://localhost:{PORTA}"
    print(f"\n  🕵️  Spy Game rodando em {url}\n  (Ctrl+C para encerrar)\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n  Missão encerrada.")
        servidor.server_close()


if __name__ == "__main__":
    main()