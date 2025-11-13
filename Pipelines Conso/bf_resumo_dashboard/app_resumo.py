import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# ==========================
# CONFIG DA PÁGINA
# ==========================
st.set_page_config(
    page_title="Alavancas Black Friday - Consolatio",
    layout="wide",
)

# ==========================
# CONSTANTES / METAS
# ==========================
METAS_FINAIS_MES = {
    "Receita":          1_288_883.24,
    "Sessoes":          140_444,
    "Investimento":     248_000.00,
    "ReceitaRetencao":  511_686.65,
    "ReceitaAquisicao": 777_196.59,
    "Pedidos":          2_479,
    "SessoesMidia":     109_735,
    "SessoesOrganicas": 30_710,
    "CAC":              142.86,
    "CPA":              100.86,
    "CPS":              1.77,
    "CPSMidia":         2.26,
    "ROAS":             5.20,
}

WEEK_SEGMENTS = [
    {"start": 1,  "end": 9,  "pct": 0.30},
    {"start": 10, "end": 16, "pct": 0.10},
    {"start": 17, "end": 23, "pct": 0.25},
    {"start": 24, "end": 30, "pct": 0.35},
]

@dataclass
class EsperadoAteHoje:
    esperado: float
    pct_ate_hoje: float  # 0..1

def _pct_esperado_ate_dia(day_of_month: int, segments=WEEK_SEGMENTS) -> float:
    pct = 0.0
    for seg in segments:
        s, e, w = seg["start"], seg["end"], seg["pct"]
        if day_of_month >= e:
            pct += w
        elif s <= day_of_month < e:
            length = (e - s + 1)
            dias_na_semana = (day_of_month - s + 1)
            pct += w * (dias_na_semana / length)
            break
    return min(max(pct, 0.0), 1.0)

def _esperado_ate_hoje_para(valor_meta_final: float, day_of_month: int) -> EsperadoAteHoje:
    pct = _pct_esperado_ate_dia(day_of_month)
    return EsperadoAteHoje(esperado=valor_meta_final * pct, pct_ate_hoje=pct)

# ==========================
# HELPERS DE FORMATAÇÃO
# ==========================
def _fmt_moeda_br(x):
    if pd.isna(x):
        return "-"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_int_br(x):
    if pd.isna(x):
        return "-"
    try:
        return f"{int(round(x)):,}".replace(",", ".")
    except Exception:
        return "-"

def _fmt_pct(x):
    if pd.isna(x):
        return "-"
    return f"{x:.1f}%"

def _highlight_delta(row):
    delta = row.get("Δ absoluto", 0)
    try:
        delta = float(delta)
    except Exception:
        delta = 0
    color = "#e6ffed" if delta >= 0 else "#ffecec"
    return [f"background-color: {color}"] * len(row)

# ==========================
# IMPORTAR BASES (CORRIGIDO PARA STREAMLIT CLOUD)
# ==========================
# APP_DIR = pasta onde está o app_resumo.py
APP_DIR   = Path(__file__).resolve().parent

# Pastas *dentro do repositório* (mesmo nível do app_resumo.py)
BASE_DIR   = APP_DIR / "Databases"
INTERM_DIR = APP_DIR / "intermediarios"

# Debug opcional (só para conferir no Cloud)
# st.write("APP_DIR:", APP_DIR)
# st.write("BASE_DIR:", BASE_DIR)
# st.write("INTERM_DIR:", INTERM_DIR)



# --- Profit-sessoes ---
profit = pd.read_csv(
    BASE_DIR / "Profit-sessoes.csv",
    sep=";",
    decimal=",",
    low_memory=False
)
# Data já vem em formato ISO (YYYY-MM-DD), parse simples
profit["Data"] = pd.to_datetime(profit["Data"], errors="coerce")

# --- Orders ---
# Importa SEM mexer em decimal (vamos tratar manualmente se precisar)
orders = pd.read_csv(
    BASE_DIR / "Orders.csv",
    sep=";",
    low_memory=False
)

# 1) Data: formato BR (dd/mm/yyyy) → precisamos forçar isso
orders["Data"] = pd.to_datetime(
    orders["Data"],
    format="%d/%m/%Y",   # IMPORTANTE: dia primeiro
    errors="coerce"
)

# 2) Faturamento liquido: detectar se está em BR (vírgula) ou EN (ponto)
col_fat = orders["Faturamento liquido"].astype(str).str.strip()

if col_fat.str.contains(",", regex=False).any():
    # Formato BR, ex: 1.234,56
    orders["Faturamento liquido"] = (
        col_fat
        .str.replace("R$", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)   # remove milhar
        .str.replace(",", ".", regex=False)  # vírgula → ponto
        .pipe(pd.to_numeric, errors="coerce")
    )
else:
    # Já está em float ou formato EN (ex: 1234.56)
    orders["Faturamento liquido"] = pd.to_numeric(
        col_fat,
        errors="coerce"
    )

# 3) Pedido válido → booleano
orders["Pedido válido"] = (
    orders["Pedido válido"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "sim", "yes"])
)

# 4) Cliente qnt de pedidos → inteiro
orders["Cliente qnt de pedidos"] = (
    orders["Cliente qnt de pedidos"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
    .fillna(0)
    .astype(int)
)

# Data como datetime (reforço)
profit["Data"] = pd.to_datetime(profit["Data"], errors="coerce")
orders["Data"] = pd.to_datetime(orders["Data"], errors="coerce")


# ==========================
# INPUT DE DATA NO LADO
# ==========================
st.sidebar.header("Filtro do período")
data_max = profit["Data"].max().date()
data_min = profit["Data"].min().date()

data_fim_custom = st.sidebar.date_input(
    "Data final do MTD",
    value=data_max,
    min_value=data_min,
    max_value=data_max,
)

# Aqui você pode futuramente adicionar data de início customizada se quiser
# por enquanto, mantemos a lógica do _kpis_mtd_basicos (MTD do mês)

from calendar import monthrange

def _primeiro_dia_mes(data):
    return data.replace(day=1)

def _dias_no_mes(data):
    return monthrange(data.year, data.month)[1]

from pandas.api import types as ptypes

def get_sum(df, col):
    """
    Soma robusta de coluna numérica (tratando vírgula/ponto somente
    quando a coluna NÃO for numérica).
    """
    if col not in df.columns:
        return 0.0

    s = df[col]

    # Se já é numérico (caso típico do Profit-sessoes), não mexe no formato
    if ptypes.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").sum()

    # Caso texto com formato BR/EN
    s = (
        s.astype(str)
         .str.replace("R$", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce").sum()


def to_float_series(s):
    """
    Converte série com formato BR/EN para float.
    Se já for numérica, só garante o tipo.
    """
    if ptypes.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    return (
        s.astype(str)
         .str.replace("R$", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
         .pipe(pd.to_numeric, errors="coerce")
    )

def _parse_money_any_locale(s):
    """
    Parser de dinheiro flexível: '1.234,56' ou '1234.56' → 1234.56 (float)
    """
    s = s.astype(str).str.strip()
    # remove símbolo de R$ se existir
    s = s.str.replace("R$", "", regex=False).str.strip()
    # casos tipo 1.234,56 (BR)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


# ==========================
# FUNÇÃO _kpis_mtd_basicos
# ==========================
def _kpis_mtd_basicos(df_profit, df_orders, data_fim: datetime) -> dict:
    """
    KPIs MTD (1º dia até data_fim):
    - Profit: Receita, Pedidos, Investimento, ROAS, CAC, TaxaConv, Ticket, CPA, Sessoes, CPS
    - Orders: ReceitaAquisicao, ReceitaRetencao (com filtro de válidos e qnt pedidos)
    """
    ini = _primeiro_dia_mes(data_fim)
    d = df_profit[(df_profit["Data"] >= ini) & (df_profit["Data"] <= data_fim)].copy()

    # Profit KPIs
    receita = get_sum(d, "Faturamento liquido")
    pedidos = int(get_sum(d, "Pedidos válidos")) if "Pedidos válidos" in d.columns else int(get_sum(d, "Pedidos"))
    investimento = get_sum(d, "Investimento em ads")
    roas = to_float_series(d["ROAS Total"]).mean() if "ROAS Total" in d.columns and len(d)>0 else 0.0
    cac = to_float_series(d["CAC"]).mean() if "CAC" in d.columns and len(d)>0 else 0.0
    taxa_conv = 0.0
    if "Taxa de conversão (%)" in d.columns and len(d)>0:
        taxa_conv = (
            d["Taxa de conversão (%)"].astype(str)
            .str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce").mean()
        )
    ticket = (receita / pedidos) if pedidos > 0 else 0.0
    cpa = (investimento / pedidos) if pedidos > 0 else 0.0

    sessoes = int(get_sum(d, "Sessões")) if "Sessões" in d.columns else 0
    cps = float(to_float_series(d["CPS"]).mean()) if "CPS" in d.columns and len(d)>0 else 0.0

    # ===============================
    # AQUISIÇÃO E RETENÇÃO — CORRIGIDO
    # ===============================
    o = df_orders[
        (df_orders["Data"] >= ini) &
        (df_orders["Data"] <= data_fim)
    ].copy()
    
    # Apenas pedidos válidos
    o_validos = o[o["Pedido válido"] == True].copy()
    
    # Separar aquisição e retenção
    o_aquis = o_validos[o_validos["Cliente qnt de pedidos"] == 1]
    o_ret   = o_validos[o_validos["Cliente qnt de pedidos"] > 1]
    
    receita_aquis = o_aquis["Faturamento liquido"].sum()
    receita_ret   = o_ret["Faturamento liquido"].sum()
    
    # Consistência: soma deve ser quase igual ao total
    total_valid = o_validos["Faturamento liquido"].sum()
    soma_ar = receita_aquis + receita_ret
    
    # Se diferença > 1%, ajuste proporcional automaticamente
    if total_valid > 0 and abs(soma_ar - total_valid) / total_valid > 0.01:
        fator = total_valid / soma_ar if soma_ar > 0 else 1
        receita_aquis *= fator
        receita_ret   *= fator


    dias_totais = _dias_no_mes(data_fim)
    dias_passados = data_fim.day

    return {
        "Receita": receita,
        "Pedidos": pedidos,
        "TicketMedio": ticket,
        "Investimento": investimento,
        "ROAS": roas,
        "CAC": cac,
        "CPA": cpa,
        "TaxaConversao": taxa_conv,
        "Sessoes": sessoes,
        "CPS": cps,
        "ReceitaAquisicao": receita_aquis,
        "ReceitaRetencao": receita_ret,
        "_tempo": {"dias_totais": dias_totais, "dias_passados": dias_passados, "ini": ini}
    }


# ==========================
# CÁLCULOS PRINCIPAIS
# ==========================
data_fim_dt = pd.to_datetime(data_fim_custom)

k = _kpis_mtd_basicos(profit, orders, data_fim_dt)
t = k["_tempo"]
dia_mes = data_fim_dt.day

# Esperados até hoje (acumulados)
exp_receita        = _esperado_ate_hoje_para(METAS_FINAIS_MES["Receita"],          dia_mes)
exp_invest         = _esperado_ate_hoje_para(METAS_FINAIS_MES["Investimento"],     dia_mes)
exp_sessoes_total  = _esperado_ate_hoje_para(METAS_FINAIS_MES["Sessoes"],          dia_mes)
exp_ret            = _esperado_ate_hoje_para(METAS_FINAIS_MES["ReceitaRetencao"],  dia_mes)
exp_aquis          = _esperado_ate_hoje_para(METAS_FINAIS_MES["ReceitaAquisicao"], dia_mes)
exp_pedidos        = _esperado_ate_hoje_para(METAS_FINAIS_MES["Pedidos"],          dia_mes)
exp_sessoes_midia  = _esperado_ate_hoje_para(METAS_FINAIS_MES["SessoesMidia"],     dia_mes)
exp_sessoes_organ  = _esperado_ate_hoje_para(METAS_FINAIS_MES["SessoesOrganicas"], dia_mes)

# ==========================
# CSV DIÁRIO DE SESSÕES
# ==========================
nome_arquivo_daily = f"sessoes_mid_org_diario_{data_fim_dt.year}-{data_fim_dt.month:02d}.csv"
df_daily = pd.read_csv(INTERM_DIR / nome_arquivo_daily, sep=";", decimal=",", low_memory=False)
df_daily["Dia"] = pd.to_datetime(df_daily["Dia"], errors="coerce").dt.date

mask_mtd = (df_daily["Dia"] >= t["ini"].date()) & (df_daily["Dia"] <= data_fim_dt.date())
d_mtd = df_daily[mask_mtd].copy()

for col in ["Sessoes_Total", "Sessoes_Midia", "Sessoes_Pagas", "Sessoes_Organicas", "CPS_Midia", "CPS_Geral", "Orcamento_Total"]:
    if col in d_mtd.columns:
        d_mtd[col] = pd.to_numeric(d_mtd[col], errors="coerce")

sess_mid_mtd = float(d_mtd["Sessoes_Midia"].sum(min_count=1) or 0.0)
sess_org_mtd = float(d_mtd["Sessoes_Organicas"].sum(min_count=1) or 0.0)
orc_mtd      = float(d_mtd["Orcamento_Total"].sum(min_count=1) or 0.0)
sess_tot_mtd_check = float(d_mtd["Sessoes_Total"].sum(min_count=1) or 0.0)

cps_midia_mtd = (orc_mtd / sess_mid_mtd) if sess_mid_mtd > 0 else 0.0
cps_geral_mtd = (orc_mtd / sess_tot_mtd_check) if sess_tot_mtd_check > 0 else 0.0

# ==========================
# MONTAR TABELAS (IGUAIS ÀS DO PNG)
# ==========================
linhas_acum = []

def _add_linha_acum(nome, real_mtd, meta_final, esperado_hoje, eh_moeda=False):
    delta_abs = real_mtd - esperado_hoje
    delta_pct = (real_mtd / esperado_hoje - 1) * 100 if esperado_hoje > 0 else np.nan
    linhas_acum.append({
        "Indicador": nome,
        "Moeda": eh_moeda,
        "Real MTD": real_mtd,
        "Esperado até hoje": esperado_hoje,
        "Δ absoluto": delta_abs,
        "Δ %": delta_pct,
        "Meta final": meta_final,
    })

# Acumulados (existentes)
_add_linha_acum("Faturamento",          k["Receita"],          METAS_FINAIS_MES["Receita"],          exp_receita.esperado,        eh_moeda=True)
_add_linha_acum("Investimento total",   k["Investimento"],     METAS_FINAIS_MES["Investimento"],     exp_invest.esperado,         eh_moeda=True)
_add_linha_acum("Receita de aquisição", k["ReceitaAquisicao"], METAS_FINAIS_MES["ReceitaAquisicao"], exp_aquis.esperado,          eh_moeda=True)
_add_linha_acum("Receita de retenção",  k["ReceitaRetencao"],  METAS_FINAIS_MES["ReceitaRetencao"],  exp_ret.esperado,            eh_moeda=True)
_add_linha_acum("Sessões (Total)",      k["Sessoes"],          METAS_FINAIS_MES["Sessoes"],          exp_sessoes_total.esperado,  eh_moeda=False)
_add_linha_acum("Pedidos",              k["Pedidos"],          METAS_FINAIS_MES["Pedidos"],          exp_pedidos.esperado,        eh_moeda=False)
_add_linha_acum("Sessões Mídia",        sess_mid_mtd,          METAS_FINAIS_MES["SessoesMidia"],     exp_sessoes_midia.esperado,  eh_moeda=False)
_add_linha_acum("Sessões Orgânicas",    sess_org_mtd,          METAS_FINAIS_MES["SessoesOrganicas"], exp_sessoes_organ.esperado,  eh_moeda=False)

df_acum = pd.DataFrame(linhas_acum)
df_acum_moeda = df_acum[df_acum["Moeda"]].drop(columns=["Moeda"]).reset_index(drop=True)
df_acum_qtd   = df_acum[~df_acum["Moeda"]].drop(columns=["Moeda"]).reset_index(drop=True)

# Tabela de médias
linhas_medias = []

def _add_linha_media(nome, media_atual, media_meta, eh_moeda=False):
    delta = media_atual - media_meta
    linhas_medias.append({
        "Indicador": nome,
        "Moeda": eh_moeda,
        "Média atual MTD": media_atual,
        "Média-meta mês": media_meta,
        "Δ": delta,
    })

_add_linha_media("CAC",                      k["CAC"],      METAS_FINAIS_MES["CAC"],      eh_moeda=True)
_add_linha_media("CPA",                      k["CPA"],      METAS_FINAIS_MES["CPA"],      eh_moeda=True)
_add_linha_media("Custo por sessão (Geral)", k["CPS"],      METAS_FINAIS_MES["CPS"],      eh_moeda=True)
_add_linha_media("Custo por sessão (Mídia)", cps_midia_mtd, METAS_FINAIS_MES["CPSMidia"], eh_moeda=True)
_add_linha_media("ROAS",                     k["ROAS"],     METAS_FINAIS_MES["ROAS"],     eh_moeda=False)

df_medias = pd.DataFrame(linhas_medias)
df_medias_moeda = df_medias[df_medias["Moeda"]].drop(columns=["Moeda"]).reset_index(drop=True)
df_medias_roas  = df_medias[~df_medias["Moeda"]].drop(columns=["Moeda"]).reset_index(drop=True)

# Checks
soma_aq_ret = k["ReceitaAquisicao"] + k["ReceitaRetencao"]
dif_pct = abs(soma_aq_ret - k["Receita"]) / k["Receita"] * 100 if k["Receita"] > 0 else 0
dif_sess = abs(sess_tot_mtd_check - k["Sessoes"]) / k["Sessoes"] * 100 if k["Sessoes"] > 0 else 0

# ==========================
# LAYOUT DO DASHBOARD
# ==========================
st.title("📊 Dashboard — Realizado vs. Projeção")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Período", f"{t['ini'].date()} → {data_fim_dt.date()}")
with col2:
    pct_mes = t['dias_passados'] / t['dias_totais'] * 100
    st.metric("Mês decorrido", f"{t['dias_passados']}/{t['dias_totais']} dias", f"{pct_mes:.1f}%")
with col3:
    st.metric("Percentual esperado até hoje", f"{exp_receita.pct_ate_hoje*100:.1f}%")

st.markdown("---")

st.subheader("💰 Acumulados (R$)")
st.dataframe(
    df_acum_moeda.style
        .apply(_highlight_delta, axis=1)
        .format({
            "Real MTD": _fmt_moeda_br,
            "Esperado até hoje": _fmt_moeda_br,
            "Δ absoluto": _fmt_moeda_br,
            "Δ %": _fmt_pct,
            "Meta final": _fmt_moeda_br,
        }),
    use_container_width=True
)

st.subheader("📈 Acumulados (Quantidade)")
st.dataframe(
    df_acum_qtd.style
        .apply(_highlight_delta, axis=1)
        .format({
            "Real MTD": _fmt_int_br,
            "Esperado até hoje": _fmt_int_br,
            "Δ absoluto": _fmt_int_br,
            "Δ %": _fmt_pct,
            "Meta final": _fmt_int_br,
        }),
    use_container_width=True
)

st.subheader("🎯 Métricas de Média (R$)")
st.dataframe(
    df_medias_moeda.style
        .apply(lambda row: _highlight_delta(row.rename({"Δ": "Δ absoluto"})), axis=1)
        .format({
            "Média atual MTD": _fmt_moeda_br,
            "Média-meta mês": _fmt_moeda_br,
            "Δ": _fmt_moeda_br,
        }),
    use_container_width=True
)

st.subheader("📊 ROAS")
st.dataframe(
    df_medias_roas.style
        .apply(lambda row: _highlight_delta(row.rename({"Δ": "Δ absoluto"})), axis=1)
        .format({
            "Média atual MTD": "{:.2f}".format,
            "Média-meta mês": "{:.2f}".format,
            "Δ": "{:.2f}".format,
        }),
    use_container_width=True
)

st.markdown("### 🔎 Checks de consistência")
st.write(f"Receita aquisição + retenção: {_fmt_moeda_br(soma_aq_ret)}")
st.write(f"Total faturado (Profit): {_fmt_moeda_br(k['Receita'])}")
st.write(f"Diferença: {dif_pct:.2f}%")
st.write(f"Sessões CSV: {_fmt_int_br(sess_tot_mtd_check)}")
st.write(f"Sessões Profit: {_fmt_int_br(k['Sessoes'])}")
st.write(f"Diferença: {dif_sess:.2f}%")
