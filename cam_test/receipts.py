"""Frozen camera bills and printable PDF receipts; no camera frames are saved."""

import copy
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

SNAPSHOT_PREFIX = 'GROCERY_SNAPSHOT:'


def freeze_bill(rows, total, unpriced, scanned_at=None):
    rows = copy.deepcopy(rows)
    calculated = missing = 0
    for name, quantity, price, amount in rows:
        if not isinstance(name, str) or type(quantity) is not int or quantity <= 0:
            raise ValueError('Invalid receipt item.')
        if price is None:
            if amount is not None:
                raise ValueError('Unpriced items cannot have an amount.')
            missing += quantity
        else:
            if type(price) is not int or price < 0 or type(amount) is not int or amount != quantity * price:
                raise ValueError('Receipt line amount does not match quantity and price.')
            calculated += amount
    if type(total) is not int or total != calculated or type(unpriced) is not int or unpriced != missing:
        raise ValueError('Receipt totals do not match its items.')
    timestamp = scanned_at or datetime.now().astimezone().isoformat(timespec='seconds')
    datetime.fromisoformat(timestamp)
    return {'rows': rows, 'total': total, 'unpriced': unpriced, 'scanned_at': timestamp}


def new_invoice(snapshot):
    bill = freeze_bill(snapshot['rows'], snapshot['total'], snapshot['unpriced'], snapshot['scanned_at'])
    if not bill['rows']:
        raise ValueError('No detected items to export. Wait for a nonempty scan.')
    now = datetime.now().astimezone()
    return {**bill, 'invoice_id': now.strftime('INV-%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8].upper(),
            'exported_at': now.isoformat(timespec='seconds')}


def write_pdf(invoice, destination):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    # Revalidate the frozen bill before writing any file.
    freeze_bill(invoice['rows'], invoice['total'], invoice['unpriced'], invoice['scanned_at'])
    font = 'Helvetica'
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/arial.ttf'
    if font_path.is_file():
        if 'ReceiptArial' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('ReceiptArial', str(font_path)))
        font = 'ReceiptArial'
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = font
    styles['Normal'].fontSize = 9
    styles['Normal'].leading = 13

    def paragraph(text, style='Normal'):
        return Paragraph(escape(str(text)), styles[style])

    story = [paragraph('GROCERYSTUFFS', 'Title'), paragraph('PRODUCT SCAN RECEIPT - DEMO', 'Heading2'),
             paragraph('Invoice: ' + invoice['invoice_id']),
             paragraph('Scan time: ' + invoice['scanned_at']),
             paragraph('Export time: ' + invoice['exported_at']), Spacer(1, 18)]
    data = [['Product', 'Qty', 'Unit (VND)', 'Amount (VND)']]
    for name, quantity, price, amount in invoice['rows']:
        data.append([paragraph(name.replace('_', ' ')), str(quantity),
                     f'{price:,}' if price is not None else 'NO PRICE',
                     f'{amount:,}' if amount is not None else 'EXCLUDED'])
    table = Table(data, colWidths=[235, 45, 105, 110], repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font), ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#173b45')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f5f6')]),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#173b45'))]))
    story.extend([table, Spacer(1, 18), paragraph(f"TOTAL: {invoice['total']:,} VND", 'Heading2'),
                  paragraph(f"Detected items: {sum(row[1] for row in invoice['rows'])}"),
                  paragraph(f"Unpriced items: {invoice['unpriced']} (excluded from total)."),
                  Spacer(1, 14), paragraph('Frozen scan snapshot. Later camera scans do not change this receipt.'),
                  paragraph('Demo receipt generated from model predictions. Please verify items and prices.')])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix='.pdf', dir=destination.parent)
    os.close(descriptor)
    try:
        doc = SimpleDocTemplate(temporary, pagesize=A4, leftMargin=50, rightMargin=50,
                                topMargin=45, bottomMargin=45, title=invoice['invoice_id'], author='Grocerystuffs')
        doc.build(story)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination
