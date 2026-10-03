
import glob
import os
import sys

from app.pipeline import DocQA

folder = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
qa = DocQA()
for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
    print(qa.ingest_pdf(path))
