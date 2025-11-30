from chunk import extract_unstructured
chunks=extract_unstructured(pdf_path="qatar_test_doc.pdf")

tables=[]
texts=[]

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    if "CompositeElement" in str(type(chunk)):
        tables.append(chunk)
    
def get_images_base64(chunks):
    images_b64=[]
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els=chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images=get_images_base64(chunks)