"""Read-only knowledge routes backed exclusively by a packaged JSON resource."""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
import re
import unicodedata
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class KnowledgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(KnowledgeModel):
    title: str
    url: str | None
    source_id: str = ''
    locator: str = ''
    original_id: str = ''


class OriginalNote(KnowledgeModel):
    locator: str
    text: str
    page: int | None = None


class OriginalVideoPart(KnowledgeModel):
    p: int
    label: str
    duration: str | None = None
    first_batch_reason: str | None = None
    note_entry_id: str | None = None
    note_title: str | None = None


class OriginalVideo(KnowledgeModel):
    access: Literal['public_video_page_playback_unverified', 'paid_course_confirm_before_use']
    bvid: str | None = None
    parts: list[OriginalVideoPart]
    first_batch_reason: str | None = None
    playback_verified: bool
    transcript_status: str
    content_review_status: str


class Original(KnowledgeModel):
    id: str
    title: str
    type: Literal['pdf', 'website', 'video']
    url: str
    embed_url: str | None = None
    source_pdf_url: str | None = None
    cover_url: str | None = None
    cover_kind: Literal['cover', 'title-page'] | None = None
    credit: str
    version: str
    license: str
    license_url: str | None = None
    notes: list[OriginalNote]
    video: OriginalVideo | None = None


class Section(KnowledgeModel):
    title: str
    paragraphs: list[str]
    bullets: list[str]
    citations: list[Citation]
    id: str = ''
    blocks: list[dict] = []


class Source(KnowledgeModel):
    id: str = ''
    title: str
    url: str | None
    locator: str
    read_scope: str
    access: str
    checked_at: str


class Summary(KnowledgeModel):
    id: str
    title: str
    kind: str
    kind_label: str
    subtype: str
    topics: list[str]
    summary: str
    authors: str
    year: int | None
    read_status_label: str
    reading_status: str = ''
    snippet: str = ''


class Entry(Summary):
    url: str | None
    sections: list[Section]
    sources: list[Source]
    related: list[Summary]
    previous: Summary | None = None
    next: Summary | None = None
    originals: list[Original] = []


class OriginalReference(KnowledgeModel):
    id: str
    title: str


class IndexedSummary(Summary):
    original_refs: list[OriginalReference] = []


class Facet(KnowledgeModel):
    id: str
    title: str
    count: int


class Overview(KnowledgeModel):
    version: str
    updated_at: str
    topics: list[Facet]
    kinds: list[Facet]
    total: int
    featured: list[Summary]
    volumes: list[dict] = []


class Results(KnowledgeModel):
    items: list[IndexedSummary]
    total: int
    page: int
    page_size: int


def normalized(value):
    return unicodedata.normalize("NFKC", value).casefold()


# Query synonyms are matched against the actual entry body, never added to
# every member of a broad topic. For example FT must not match all processing.
SYNONYMS = [
    ("小波", "小波变换", "wavelet", "wavelet transform", "wt", "wt-exafs"),
    ("傅里叶", "傅里叶变换", "fourier", "fourier transform", "ft"),
    ("归一化", "normalization", "normalisation"),
    ("荧光", "fluorescence"), ("透射", "transmission"),
    ("配位数", "coordination number"), ("自吸收", "self-absorption", "过吸收", "over-absorption"),
    ("原位", "in situ", "operando", "操作态"),
    ("线性组分", "linear combination", "lcf"), ("主成分", "principal component", "pca"),
    # Retrieval of the MCR family does not equate its regression algorithms.
    ("多元曲线", "多元曲线分辨", "multivariate curve", "mcr", "mcr-als", "mcr-ar"),
    ("不确定度", "uncertainty", "uncertainties"),
    ("sha-256", "sha256"),
]


def contains(text, phrase):
    # Short ASCII abbreviations should match a word, not WT in an unrelated URL
    # or N in every English title. URLs and provenance are not search content.
    if phrase.isascii() and phrase.isalnum() and len(phrase) <= 4:
        return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None
    return phrase in text


@lru_cache(maxsize=1)
def catalog():
    snapshot = json.loads(files("xasweb").joinpath("knowledge", "catalog.json").read_text(encoding="utf-8"))
    entries = snapshot["entries"]
    topic_names = {topic["id"]: topic["title"] for topic in snapshot["topics"]}
    indexed = {}
    for item in entries:
        # Review/rights boilerplate and common source titles must not make
        # unrelated entries appear to discuss a topic.
        content = [item["title"], item["summary"], item["authors"], *item.get("aliases", []),
                   *(topic_names[topic] for topic in item["topics"])]
        for part in item["sections"]:
            if part["title"] not in {"阅读范围", "收录与访问边界", "验证与证据边界", "版本与书目信息"}:
                content.extend(part["paragraphs"])
                content.extend(part["bullets"])
                for block in part.get('blocks', []):
                    content.extend([block.get('text',''),block.get('caption',''),*block.get('items',[]),*block.get('headers',[])])
                    for row in block.get('rows',[]):content.extend(row)
        indexed[item["id"]] = normalized("\n".join(content))
    return snapshot, {item["id"]: item for item in entries}, indexed


def summary(item):
    return {name: item.get(name, '') for name in Summary.model_fields}


@router.get("", response_model=Overview)
def overview():
    snapshot, _, _ = catalog()
    entries = snapshot["entries"]
    return {"version": snapshot["version"], "updated_at": snapshot["updated_at"],
            "topics": [{**topic, "count": sum(topic["id"] in item["topics"] for item in entries)}
                       for topic in snapshot["topics"]],
            "kinds": [{**kind, "count": sum(kind["id"] == item["kind"] for item in entries)}
                      for kind in snapshot["kinds"]],
            "total": len(entries), "volumes": snapshot.get('volumes', []),
            "featured": [summary(item) for ident in snapshot.get('featured_ids',[]) for item in entries if item['id']==ident]}


@router.get("/entries", response_model=Results)
def list_entries(q: str = Query(default="", max_length=200),
                 topic: str = Query(default="", max_length=40),
                 kind: str = Query(default="", max_length=40),
                 page: int = Query(default=1, ge=1),
                 page_size: int = Query(default=12, ge=1, le=60),
                 view: str = Query(default='', pattern='^(|library|atlas)$')):
    snapshot, _, indexed = catalog()
    topic, kind = topic.strip().casefold(), kind.strip().casefold()
    if topic and topic not in {item["id"] for item in snapshot["topics"]}:
        raise HTTPException(status_code=422, detail="未知专题")
    if kind and kind not in {item["id"] for item in snapshot["kinds"]}:
        raise HTTPException(status_code=422, detail="未知资料类型")
    query = normalized(q).strip()
    # Preserve a known multiword synonym as one concept, otherwise combine
    # whitespace-separated terms with AND, as with the topic/kind filters.
    phrases = [query] if any(query in group for group in SYNONYMS) else query.split()
    matches = []
    for item in snapshot["entries"]:
        if view=='library' and item['kind'] not in {'literature','blog','manual','video','video_note'}:continue
        if view=='atlas' and item['kind'] not in {'case','figure'}:continue
        if topic and topic not in item["topics"] or kind and kind != item["kind"]:
            continue
        body = indexed[item["id"]]
        groups = [next((group for group in SYNONYMS if phrase in group), (phrase,)) for phrase in phrases]
        if all(any(contains(body, alternative) for alternative in group) for group in groups):
            matches.append(item)
    if query:
        def rank(item):
            title=normalized(item['title']);aliases=normalized(' '.join(item.get('aliases',[])))
            return (-(sum(8*contains(title,p)+4*contains(aliases,p) for p in phrases)+(2 if item['kind']=='guide' else 0)),item['id'])
        matches.sort(key=rank)
    offset = (page - 1) * page_size
    originals_by_entry = {}
    if view == 'library':
        for original in snapshot.get('originals', []):
            reference = {'id': original['id'], 'title': original['title']}
            for entry_id in original['entry_ids']:
                originals_by_entry.setdefault(entry_id, []).append(reference)
    results=[]
    for item in matches[offset:offset+page_size]:
        record=summary(item)
        if item['id'] in originals_by_entry:
            record['original_refs'] = originals_by_entry[item['id']]
        if query:
            body=indexed[item['id']];positions=[body.find(p) for p in phrases if p in body];pos=min(positions) if positions else 0
            record['snippet']=body[max(0,pos-35):pos+130].replace('\n',' ')
        results.append(record)
    return {"items": results,
            "total": len(matches), "page": page, "page_size": page_size}


@router.get("/entries/{entry_id}", response_model=Entry)
def entry_detail(entry_id: str):
    snapshot, index, _ = catalog()
    item = index.get(entry_id)
    if item is None:
        raise HTTPException(status_code=404, detail="未找到该知识条目")
    topics = set(item["topics"])
    candidates = [other for other in index.values()
                  if other["id"] != entry_id and topics.intersection(other["topics"])]
    candidates.sort(key=lambda other: (-len(topics.intersection(other["topics"])), other["kind"] != "guide"))
    previous=following=None
    for volume in snapshot.get('volumes',[]):
        order=[c['id'] for c in volume['chapters'] if c['id'] in index]
        if entry_id in order:
            at=order.index(entry_id)
            previous=summary(index[order[at-1]]) if at else None
            following=summary(index[order[at+1]]) if at+1<len(order) else None
    # Learning routes connect the end of one volume to the start of the next topic.
    for chapter_route in (('guide-G01', 'guide-G02', 'guide-G05'),
                          ('guide-G22', 'guide-G23', 'guide-G24')):
        available_route = [ident for ident in chapter_route if ident in index]
        if entry_id in available_route:
            at = available_route.index(entry_id)
            if previous is None and at:
                previous = summary(index[available_route[at - 1]])
            if following is None and at + 1 < len(available_route):
                following = summary(index[available_route[at + 1]])
    cited_originals = {citation.get('original_id') for part in item['sections'] for citation in part['citations']}
    return {**summary(item), "url": item["url"], "sections": item["sections"], "sources": item["sources"],
            "previous":previous,"next":following,"related": [summary(other) for other in candidates[:3]],
            "originals": [public_original(row) for row in snapshot.get('originals', [])
                          if entry_id in row['entry_ids'] or row['id'] in cited_originals]}


def public_original(row):
    result = {key: row[key] for key in Original.model_fields if key in row}
    if row.get('file'):
        result['embed_url'] = f"/api/knowledge/originals/{row['id']}/pdf"
        if cover := row.get('cover'):
            result['cover_url'] = f"/api/knowledge/originals/{row['id']}/cover"
            result['cover_kind'] = cover['kind']
    return result


def original_record(original_id):
    snapshot, _, _ = catalog()
    row = next((row for row in snapshot.get('originals', []) if row['id'] == original_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="未找到该原始资料")
    return row


@router.get('/originals', response_model=list[Original])
def originals():
    snapshot, _, _ = catalog()
    return [public_original(row) for row in snapshot.get('originals', [])]


@router.get('/originals/{original_id}', response_model=Original)
def original_detail(original_id: str):
    return public_original(original_record(original_id))


@router.get('/originals/{original_id}/pdf')
def original_pdf(original_id: str):
    row = original_record(original_id)
    filename = row.get('file', '')
    if row['type'] != 'pdf' or not re.fullmatch(r'[a-z][a-z0-9-]+\.pdf', filename):
        if row['type'] == 'website' and re.fullmatch(r'https://doi\.org/10\.\d{4,9}/[^?#\s]+', row['url']):
            return RedirectResponse(row['url'], status_code=302)
        raise HTTPException(status_code=404, detail="该资料没有馆内 PDF")
    path = files('xasweb').joinpath('knowledge', 'assets', filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="原文文件暂不可用")
    return FileResponse(
        str(path), media_type='application/pdf', filename=filename, content_disposition_type='inline',
        headers={'Cache-Control': 'public, max-age=3600', 'X-Content-Type-Options': 'nosniff',
                 'X-Frame-Options': 'SAMEORIGIN'},
    )


@router.get('/originals/{original_id}/cover')
def original_cover(original_id: str):
    row = original_record(original_id)
    filename = row.get('cover', {}).get('file')
    if row['type'] != 'pdf' or not row.get('file') or filename != f'{original_id}-cover.png':
        raise HTTPException(status_code=404, detail='该资料没有原封面')
    path = files('xasweb').joinpath('knowledge', 'assets', filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail='封面暂不可用')
    return FileResponse(str(path), media_type='image/png',
                        headers={'Cache-Control': 'public, max-age=3600', 'X-Content-Type-Options': 'nosniff'})


@router.get('/assets/{asset_id}')
def knowledge_asset(asset_id: str):
    snapshot,_,_=catalog()
    asset=next((a for a in snapshot.get('assets',[]) if a['id']==asset_id),None)
    if asset is None or not re.fullmatch(r'[A-Za-z0-9_.-]+',asset['file']):
        raise HTTPException(status_code=404,detail='未找到图件')
    path=files('xasweb').joinpath('knowledge','assets',asset['file'])
    if not path.is_file():raise HTTPException(status_code=404,detail='图件暂不可用')
    return FileResponse(str(path),headers={'Cache-Control':'public, max-age=3600','X-Content-Type-Options':'nosniff'})
