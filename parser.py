#!/usr/bin/env python3
# parser.py

"""
Parse plain-ICAO NOTAM .txt files under raw_notam_text/ into a list of
structured record dicts, ready for ingestion (no JSON file output).
"""

import sys, uuid, datetime, re
from pathlib import Path
import importlib.util
from types import SimpleNamespace
from Qcode_mapping import SUBJECT_MAP, CONDITION_MAP, STATUS_MAP

# ─── Dynamically load PyNotam & ICAO abbreviations ────────────────────────────
BASE = Path(__file__).parent
PNP  = BASE / "PyNotam"
sys.path.insert(0, str(PNP))

_spec_notam = importlib.util.spec_from_file_location("notam_mod", str(PNP / "notam.py"))
_notam_mod  = importlib.util.module_from_spec(_spec_notam)
_spec_notam.loader.exec_module(_notam_mod)
Notam       = _notam_mod.Notam

_spec_abbr  = importlib.util.spec_from_file_location("abbr_mod", str(PNP / "_abbr.py"))
_abbr_mod   = importlib.util.module_from_spec(_spec_abbr)
_spec_abbr.loader.exec_module(_abbr_mod)
ICAO_abbr   = _abbr_mod.ICAO_abbr

# ─── Abbreviation expansion ────────────────────────────────────────────────────
_ABBR_PATTERN = re.compile(r"\b(" + "|".join(
    re.escape(k) for k in sorted(ICAO_abbr.keys(), key=len, reverse=True)
) + r")\b")
def expand_abbr(text: str) -> str:
    return _ABBR_PATTERN.sub(lambda m: ICAO_abbr[m.group(1)], text)

# ─── Timestamp helpers ─────────────────────────────────────────────────────────
def parse_created(raw: str) -> str:
    m = re.search(r"CREATED:\s*([0-3]\d\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2})", raw)
    if not m: return ""
    try:
        dt = datetime.datetime.strptime(m.group(1), "%d %b %Y %H:%M:%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    except ValueError:
        return ""

def format_iso(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z") if dt else ""

# ─── Text normalization ────────────────────────────────────────────────────────
CLAUSE_LETTERS = "QABCDEFG"
def normalize(raw: str) -> str:
    text = raw.replace("\r", "").strip()
    if not text.startswith("("): text = "(" + text
    if not text.endswith(")"):   text = text + ")"
    text = re.sub(rf"\s*([{CLAUSE_LETTERS}])\)", r"\n\1)", text)
    text = re.sub(rf"([{CLAUSE_LETTERS}])\)([^\s])", r"\1) \2", text)
    return text

# ─── Fallback parser for unparseable NOTAMs ────────────────────────────────────
def fallback_parse(text: str):
    lines = text.splitlines()
    if not lines: return None
    rec = {}; body_lines = []
    m_id = re.match(r"([A-Z]\d{3,4}/\d{2})", lines[0].strip())
    rec['notam_id'] = m_id.group(1) if m_id else ''
    for ln in lines[1:]:
        txt = ln.strip()
        if txt.startswith('Q)'):
            part = txt.split(')',1)[1].strip()
            m = re.match(r"(Q[A-Z]{4})", part)
            rec['notam_code'] = m.group(1) if m else ''
        elif txt.startswith('A)'):
            rec['location'] = txt[2:].strip()
        elif txt.startswith('B)'):
            rec['B'] = txt[2:].strip()
        elif txt.startswith('C)'):
            rec['C'] = txt[2:].strip()
        elif txt.startswith(('E)','F)','G)')):
            parts = txt.split(')',1)
            body_lines.append(parts[1].strip() if len(parts)>1 else '')
        elif body_lines:
            body_lines.append(txt)
    vf = vt = None
    if rec.get('B','').isdigit():
        try: vf = datetime.datetime.strptime(rec['B'],'%y%m%d%H%M')
        except: pass
    if rec.get('C','').isdigit():
        try: vt = datetime.datetime.strptime(rec['C'],'%y%m%d%H%M')
        except: pass
    return {
        'notam_id': rec.get('notam_id',''),
        'notam_code': rec.get('notam_code',''),
        'body': '\n'.join(body_lines).strip(),
        'valid_from': vf,
        'valid_till': vt,
        'location': rec.get('location',''),
        'country': ''
    }

# ─── Main parser ──────────────────────────────────────────────────────────────
def parse_all_notams(src_dir: Path) -> list[dict]:
    records = []
    seen_missing = {'subject': set(), 'condition': set(), 'modifier': set()}
    seen_bad_locs = set()

    for fp in sorted(Path(src_dir).glob("*.txt")):
        raw  = fp.read_text(encoding="utf-8", errors="ignore")
        norm = normalize(raw)
        try:
            n = Notam.from_str(norm)
        except:
            fb = fallback_parse(norm)
            if not fb:
                print(f"[WARN] Skipping {fp.name}")
                continue
            n = SimpleNamespace(**fb)

        # ID and Q-code
        id_val = getattr(n,'notam_id','') or ''
        if not id_val:
            m2 = re.search(r"([A-Z]\d{3,4}/\d{2})", raw)
            if m2: id_val = m2.group(1)
        m_code = re.search(r"(Q[A-Z]{4})", norm)
        if m_code: setattr(n,'notam_code',m_code.group(1))

        # Dates
        start_dt = getattr(n,'valid_from',None)
        end_dt   = getattr(n,'valid_till',None)
        if not start_dt or not end_dt:
            m = re.search(r"B\)\s*(\d{10}).*?C\)\s*(\d{10})", norm)
            if m:
                try:
                    start_dt = datetime.datetime.strptime(m.group(1),'%y%m%d%H%M')
                    end_dt   = datetime.datetime.strptime(m.group(2),'%y%m%d%H%M')
                except: pass

        created = parse_created(raw)
        code    = getattr(n,'notam_code','') or ''
        entity  = code[1:3] if len(code)>=5 else ''
        status  = code[3:5] if len(code)>=5 else ''
        qcode   = code[1:] if code.startswith('Q') else code

        # Area/SubArea
        subj = code[1:3] if len(code)>=3 else ''
        tup  = SUBJECT_MAP.get(subj)
        if tup and len(tup)>=2:
            area, subarea = tup[0], tup[1].replace('&','and')
        else:
            area, subarea = 'not defined','not defined'
            if subj not in seen_missing['subject']:
                print(f"[WARN] SUBJECT_MAP missing key: '{subj}'")
                seen_missing['subject'].add(subj)

        # Condition & Modifier
        ck = status[:1]
        condition = CONDITION_MAP.get(ck,'not defined')
        if ck not in CONDITION_MAP and ck not in seen_missing['condition']:
            print(f"[WARN] CONDITION_MAP missing letter: '{ck}'")
            seen_missing['condition'].add(ck)
        modifier = STATUS_MAP.get(status,'not defined')
        if status not in STATUS_MAP and status not in seen_missing['modifier']:
            print(f"[WARN] STATUS_MAP missing code: '{status}'")
            seen_missing['modifier'].add(status)

        # Location A)
        m_loc = re.search(r"\bA\)\s*([^\n]+)", norm)
        raw_a = (m_loc.group(1).strip() if m_loc else '')
        raw_a = re.sub(r"^\(REF:[^)]+\)", "", raw_a)
        toks  = re.findall(r"\b[A-Z]{4}\b", raw_a)
        if toks:
            loc, is_icao = toks[0], True
        else:
            loc, is_icao = "", False
            if raw_a and raw_a not in seen_bad_locs:
                print(f"[WARN] A) clause invalid ICAO: '{raw_a}'")
                seen_bad_locs.add(raw_a)

        body = getattr(n,'body','').strip()

        rec = {
            '_id': uuid.uuid4().hex,
            'id': id_val,
            'entity': entity,
            'status': status,
            'Qcode': qcode,
            'Area': area,
            'SubArea': subarea,
            'Condition': condition,
            'Subject': '',
            'Modifier': modifier,
            'message': body,
            'message_expanded': expand_abbr(body),
            'all_expanded': expand_abbr(raw),
            'startdate': format_iso(start_dt),
            'enddate': format_iso(end_dt),
            'all': raw.strip(),
            'location': loc,
            'isICAO': is_icao,
            'Created': created,
            'key': f"{id_val}-{loc}".strip("- "),
            'type': 'airport',
            'quality': {},
            'StateCode': getattr(n,'country','') or '',
            'StateName': ''
        }
        records.append(rec)

    return records
