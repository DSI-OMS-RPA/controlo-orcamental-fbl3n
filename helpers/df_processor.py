import pandas as pd
from pathlib import Path
import chardet
import re
import unicodedata
from datetime import datetime

from helpers.utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------
# ENCODING DETECTION
# ---------------------------------------------------------
def detect_encoding(file_path: Path, sample_size: int = 8192) -> str:
    """Detecta encoding com chardet."""
    with open(file_path, "rb") as f:
        raw = f.read(sample_size)

    result = chardet.detect(raw)
    enc = result.get("encoding")
    conf = result.get("confidence", 0)

    if enc:
        logger.info(f"Encoding detectado: {enc} ({conf:.2f})")
        return enc

    return "utf-8"


def ensure_utf8(file_path: Path) -> Path:
    """Converte UTF-16 para UTF-8 quando necessário."""
    encoding = detect_encoding(file_path)

    if encoding and encoding.lower().startswith("utf-16"):
        out_path = file_path.with_suffix(".utf8.txt")
        logger.info("Convertendo UTF-16 para UTF-8...")

        with open(file_path, "r", encoding=encoding, errors="replace") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            fout.write(fin.read())

        logger.info(f"Conversao completa: {out_path}")
        return out_path

    return file_path


# ---------------------------------------------------------
# DATA CLEANING HELPERS
# ---------------------------------------------------------
def clean_column_name(name: str) -> str:
    """
    Normaliza nomes de colunas para snake_case sem caracteres especiais.
    Trata acentos e caracteres Unicode de forma segura.
    """
    try:
        # Convert to string
        s = str(name).strip()

        # Normalize Unicode (decompose characters like á -> a + combining accent)
        s = unicodedata.normalize('NFKD', s)

        # Remove accents and diacritics
        s = ''.join(c for c in s if not unicodedata.combining(c))

        # Replace multiple spaces with single space
        s = re.sub(r"\s+", " ", s)

        # Replace non-alphanumeric chars (except underscore) with underscore
        # This is safer than the original regex which had invalid character range
        s = re.sub(r"[^a-z0-9_\s]", "_", s.lower())

        # Replace spaces with underscore
        s = re.sub(r"\s+", "_", s)

        # Replace multiple underscores with single underscore
        s = re.sub(r"_+", "_", s)

        # Remove leading/trailing underscores
        return s.strip("_")

    except Exception as e:
        logger.warning(f"Error cleaning column name '{name}': {e}")
        return "column"


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza a nomes de colunas."""
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas que parecem datas usando coercion segura.
    Evita deprecacao do errors='ignore' substituindo por errors='coerce'.
    """
    date_cols = [
        c for c in df.columns
        if any(hint in c.lower() for hint in ['dt', 'data', 'entrada'])
    ]

    for col in date_cols:
        try:
            # Usar coerce para converter valores invalidos em NaT
            # dayfirst=True para formato europeu (DD/MM/YYYY)
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            logger.debug(f"Coluna '{col}' convertida para datetime")
        except Exception as e:
            logger.warning(f"Nao foi possivel converter coluna '{col}': {str(e)}")

    return df


# ---------------------------------------------------------
# MAIN FUNCTION (RETORNA DATAFRAME)
# ---------------------------------------------------------
def load_partidas_file(file_path: str | Path, skiprows: int = 5) -> pd.DataFrame:
    """
    Carrega e processa o ficheiro export_partidas_aberto.
    Retorna um pandas DataFrame limpo e normalizado.

    Args:
        file_path (str | Path): Caminho do ficheiro
        skiprows (int): Numero de linhas a pular (default: 5)

    Returns:
        pd.DataFrame: DataFrame processado e normalizado

    Raises:
        AssertionError: Se o ficheiro nao existir
        Exception: Se houver erro na leitura/processamento
    """
    file_path = Path(file_path)
    assert file_path.exists(), f"Ficheiro nao encontrado: {file_path}"

    # Garantir encoding UTF-8
    file_path = ensure_utf8(file_path)

    logger.info(f"Lendo ficheiro TAB-delimited: {file_path}")

    try:
        # Ler como TAB-separated com engine python para melhor compatibilidade
        df = pd.read_csv(
            file_path,
            sep="\t",
            skiprows=skiprows,
            dtype=str,
            engine="python",
            encoding="utf-8"
        )

        # Remover linhas totalmente vazias
        df = df.dropna(how="all")

        # Normalizar nomes das colunas (SAFE: sem regex invalido)
        df = normalize_column_names(df)

        # Remover colunas sem nome
        df = df.loc[:, ~df.columns.str.startswith("unnamed")]

        # Normalizar datas com coerce (seguro)
        df = normalize_dates(df)

        # Log resumo
        logger.info(f"Linhas: {len(df)} | Colunas: {len(df.columns)}")
        logger.debug(f"Colunas: {list(df.columns)}")

        return df

    except pd.errors.ParserError as e:
        logger.error(f"Erro ao fazer parse do ficheiro: {str(e)}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Erro de encoding ao ler ficheiro: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao processar ficheiro: {str(e)}")
        raise
