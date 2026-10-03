


def extract_pdf(path: str):
    """Return a list of (page_number, text). Page numbers start at 1."""
    import fitz  

    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = " ".join(page.get_text("text").split())  
            if text:
                pages.append((i, text))
    return pages


def chunk_pages(doc_id: str, pages, size: int = 200, overlap: int = 40):
    
    step = max(1, size - overlap)
    chunks = []
    for page_no, text in pages:
        words = text.split()
        for ci, start in enumerate(range(0, len(words), step)):
            piece = words[start:start + size]
            chunks.append({
                "chunk_id": f"{doc_id}::p{page_no}::c{ci}",
                "doc_id": doc_id,
                "page": page_no,
                "text": " ".join(piece),
            })
            if start + size >= len(words): 
                break
    return chunks
