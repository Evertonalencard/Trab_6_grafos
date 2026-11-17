"""
Spy Game — núcleo algorítmico.

Implementa:
  * Bellman-Ford com registro passo a passo (para a interface mostrar cada iteração);
  * detecção de ciclos de peso negativo e extração dos vértices que formam o ciclo;
  * identificação dos esconderijos "contaminados" pelo ciclo (custo -inf).

Nenhuma dependência externa: só a biblioteca padrão do Python.
"""

from __future__ import annotations

from collections import deque

INF = float("inf")


# ---------------------------------------------------------------------------
# Bellman-Ford
# ---------------------------------------------------------------------------
def bellman_ford(vertices, edges, source):
    """
    vertices: lista de nomes de vértices, ex. ["S", "A", "B"]
    edges:    lista de tuplas (origem, destino, peso)
    source:   vértice de origem

    Retorna um dicionário com distâncias, predecessores, a execução passo a passo,
    os ciclos negativos encontrados e os vértices afetados por eles.
    """
    n = len(vertices)
    dist = {v: INF for v in vertices}
    pred = {v: None for v in vertices}
    dist[source] = 0

    iteracoes = []

    # Fase 1: |V| - 1 rodadas de relaxamento de TODAS as arestas.
    for i in range(1, n):
        relaxadas = []
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                relaxadas.append(
                    {"origem": u, "destino": v, "peso": w,
                     "antes": dist[v], "depois": dist[u] + w}
                )
                dist[v] = dist[u] + w
                pred[v] = u
        iteracoes.append({"iteracao": i, "relaxacoes": relaxadas, "dist": dict(dist)})
        if not relaxadas:  # parada antecipada: nada mudou, já convergiu
            break

    # Fase 2: rodada extra (a |V|-ésima). Se ainda houver relaxamento,
    # existe ciclo negativo alcançável a partir da origem.
    dist_v = dict(dist)
    pred_v = dict(pred)
    arestas_violadas = []
    vertices_relaxados = []
    for u, v, w in edges:
        if dist_v[u] != INF and dist_v[u] + w < dist_v[v]:
            arestas_violadas.append({"origem": u, "destino": v, "peso": w,
                                     "antes": dist_v[v], "depois": dist_v[u] + w})
            dist_v[v] = dist_v[u] + w
            pred_v[v] = u
            vertices_relaxados.append(v)

    # Fase 3: recuperar os ciclos andando pelos predecessores.
    ciclos = []
    vistos = set()
    for x in vertices_relaxados:
        ciclo = _extrair_ciclo(x, pred_v, n)
        if ciclo:
            chave = frozenset(ciclo)
            if chave not in vistos:
                vistos.add(chave)
                ciclos.append(ciclo)

    # Fase 4: tudo que é alcançável a partir de um ciclo negativo tem custo -inf.
    adj = _adjacencia(vertices, edges)
    sementes = {v for c in ciclos for v in c} | set(vertices_relaxados)
    afetados = _alcancaveis(sementes, adj)

    final = {}
    for v in vertices:
        if v in afetados:
            final[v] = -INF
        else:
            final[v] = dist[v]

    return {
        "dist": final,
        "pred": pred,
        "iteracoes": iteracoes,
        "verificacao": arestas_violadas,
        "ciclos": [
            {"vertices": c, "peso": _peso_ciclo(c, edges)} for c in ciclos
        ],
        "afetados": sorted(afetados, key=vertices.index),
        "tem_ciclo_negativo": bool(arestas_violadas),
    }


def _extrair_ciclo(x, pred, n):
    """
    Partindo de um vértice relaxado na |V|-ésima rodada, volta |V| vezes pelos
    predecessores. Depois de |V| passos é garantido estar DENTRO do ciclo
    (o caminho de predecessores tem no máximo |V| vértices distintos).
    Daí seguimos os predecessores até voltar ao mesmo vértice.
    """
    for _ in range(n):
        if pred[x] is None:
            return None
        x = pred[x]
    ciclo = [x]
    y = pred[x]
    while y != x:
        ciclo.append(y)
        y = pred[y]
    ciclo.reverse()  # predecessores dão a ordem inversa; invertendo, fica na ordem das arestas
    # normaliza para começar pelo menor rótulo (fica estável na interface)
    k = ciclo.index(min(ciclo))
    return ciclo[k:] + ciclo[:k]


def _peso_ciclo(ciclo, edges):
    pesos = {}
    for u, v, w in edges:
        pesos[(u, v)] = min(w, pesos.get((u, v), INF))
    total = 0
    for i, u in enumerate(ciclo):
        v = ciclo[(i + 1) % len(ciclo)]
        total += pesos[(u, v)]
    return total


def _adjacencia(vertices, edges):
    adj = {v: [] for v in vertices}
    for u, v, _ in edges:
        adj[u].append(v)
    return adj


def _alcancaveis(sementes, adj):
    vistos = set(sementes)
    fila = deque(sementes)
    while fila:
        u = fila.popleft()
        for v in adj[u]:
            if v not in vistos:
                vistos.add(v)
                fila.append(v)
    return vistos


def reconstruir_rota(pred, source, destino):
    """Caminho origem -> destino seguindo os predecessores (só para custo finito)."""
    rota = []
    atual = destino
    passos = 0
    while atual is not None:
        rota.append(atual)
        if atual == source:
            return list(reversed(rota))
        atual = pred[atual]
        passos += 1
        if passos > len(pred):
            return None
    return None


# ---------------------------------------------------------------------------
# Dados do trabalho
# ---------------------------------------------------------------------------
VERTICES = ["S", "A", "B", "C", "D", "E"]
ARESTAS_ORIGINAIS = [
    ("S", "A", 4),
    ("S", "B", 5),
    ("A", "B", -2),
    ("A", "C", 6),
    ("B", "C", 3),
    ("B", "D", 7),
    ("C", "D", 2),
]
ARESTA_ARMADILHA = ("D", "B", -6)


def _fmt(x):
    if x == INF:
        return "∞"
    if x == -INF:
        return "-∞"
    return str(x)


def _relatorio(titulo, vertices, edges, source="S"):
    r = bellman_ford(vertices, edges, source)
    print("=" * 64)
    print(titulo)
    print("=" * 64)
    for it in r["iteracoes"]:
        rel = ", ".join(f"{x['origem']}→{x['destino']}" for x in it["relaxacoes"]) or "nenhuma"
        dists = "  ".join(f"{v}={_fmt(it['dist'][v])}" for v in vertices)
        print(f"Iteração {it['iteracao']}: {dists}   | relaxadas: {rel}")
    print()
    for v in vertices:
        d = r["dist"][v]
        if d == -INF:
            status = "ARMADILHA (afetado por ciclo negativo)"
            rota = "—"
        elif d == INF:
            status = "inalcançável"
            rota = "—"
        else:
            status = "ok"
            rota = " → ".join(reconstruir_rota(r["pred"], source, v))
        print(f"  {v}: custo {_fmt(d):>3}   rota: {rota:<18} {status}")
    if r["ciclos"]:
        for c in r["ciclos"]:
            cam = " → ".join(c["vertices"] + [c["vertices"][0]])
            print(f"\n  Ciclo negativo: {cam}  (peso total = {c['peso']})")
        print(f"  Esconderijos-armadilha: {', '.join(r['afetados'])}")
    else:
        alc = [(v, r["dist"][v]) for v in vertices if v != source and r["dist"][v] not in (INF, -INF)]
        menor = min(alc, key=lambda t: t[1])
        maior = max(alc, key=lambda t: t[1])
        print(f"\n  Menor risco: {menor[0]} ({menor[1]})   Maior risco: {maior[0]} ({maior[1]})")
    print()


if __name__ == "__main__":
    _relatorio("Experimento 1 — Rede original", VERTICES, ARESTAS_ORIGINAIS)
    _relatorio("Experimento 2 — Com a aresta D → B (-6)", VERTICES,
               ARESTAS_ORIGINAIS + [ARESTA_ARMADILHA])