def generate_pdf_report(path: str, metrics: dict) -> str:
    # TODO: implement with reportlab or weasyprint
    with open(path, "w") as f:
        f.write("PDF REPORT\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    return path
