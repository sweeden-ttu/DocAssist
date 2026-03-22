# IRS Form Documentation

## Overview

DocAssist supports extraction of IRS tax forms using Docling for document parsing and Vision-Language Models for field detection.

## Extracted Forms

### Form 1040 - U.S. Individual Income Tax Return
**Status**: ✅ Extracted with Docling

| Property | Value |
|----------|-------|
| Pages | 2 |
| Text Elements | 1,726 |
| Tables | 0 |
| Form Items | 0 |
| Size | 1.0 MB |
| Output | `f1040_docling.json` |

**Sections:**
- Filing Status
- Personal Information
- Income (Wages, Interest, Dividends, etc.)
- Adjusted Gross Income
- Standard Deduction
- Taxable Income
- Tax and Credits
- Payments
- Refund/Amount Owed

### Form W-4 - Employee's Withholding Certificate
**Status**: ✅ Extracted with Docling

| Property | Value |
|----------|-------|
| Pages | 5 |
| Text Elements | 343 |
| Tables | 3 |
| Form Items | 0 |
| Size | 1.0 MB |

**Steps:**
1. Personal Information
2. Multiple Jobs or Spouse Works
3. Claim Dependents
4. Other Adjustments (optional)
5. Signature

### Form W-9 - Request for Taxpayer ID
**Status**: ✅ Extracted with Docling

| Property | Value |
|----------|-------|
| Pages | 6 |
| Text Elements | 237 |
| Tables | 4 |
| Size | 302 KB |

**Fields:**
- Name
- Business Name
- Federal Tax Classification
- Exemptions
- Address
- SSN or EIN

### Form 1065 - Partnership Return
**Status**: ✅ Extracted with Docling

| Property | Value |
|----------|-------|
| Pages | 6 |
| Text Elements | ~1,800 |
| Tables | 1 |
| Size | 1.3 MB |

**Components:**
- Partnership Information
- Total Income
- Total Deductions
- Schedule K Items
- Partner Capital Accounts

## Form Schemas

### Form 1040 Field Schema

```json
{
  "form_type": "IRS Form 1040",
  "version": "2025",
  "fields": [
    {
      "id": "f1040_1a",
      "type": "text_input",
      "label": "First name and middle initial",
      "section": "Step 1: Personal Information",
      "fillable": true
    },
    {
      "id": "f1040_1b", 
      "type": "text_input",
      "label": "Last name",
      "section": "Step 1: Personal Information",
      "fillable": true
    },
    {
      "id": "f1040_1c",
      "type": "ssn",
      "label": "Your SSN",
      "section": "Step 1: Personal Information",
      "fillable": true
    },
    {
      "id": "f1040_2a",
      "type": "checkbox",
      "label": "Single",
      "section": "Filing Status",
      "fillable": true
    },
    {
      "id": "f1040_2b",
      "type": "checkbox",
      "label": "Married filing jointly",
      "section": "Filing Status",
      "fillable": true
    }
  ]
}
```

## Installation

```bash
conda activate taxenv
pip install docling docling-ibm-models
```

## Usage

### Extract Form with Docling
```bash
python docling_cli.py parse form1040.pdf --format json --output f1040.json
```

### Extract with GUI
```bash
python src/gui_viewer.py --image form1040.png --json f1040.json
```

## Common IRS Forms

| Form | Name | Used For |
|------|------|----------|
| 1040 | U.S. Individual Income Tax Return | Main personal tax form |
| 1040-SR | U.S. Tax Return for Seniors | Seniors version of 1040 |
| W-4 | Employee's Withholding Certificate | Tax withholding |
| W-9 | Request for Taxpayer ID | Tax ID requests |
| 1065 | U.S. Return of Partnership Income | Partnership taxes |
| 1120 | U.S. Corporation Income Tax Return | Corporate taxes |
| 941 | Employer's Quarterly Federal Tax Return | Payroll taxes |

## Schedules for Form 1040

| Schedule | Name | When Required |
|----------|------|---------------|
| 1 | Additional Income and Adjustments | If you have other income |
| 2 | Additional Taxes | If you owe AMT or other taxes |
| 3 | Additional Credits and Payments | If you have foreign tax, etc. |
| A | Itemized Deductions | If you itemize instead of standard deduction |
| B | Interest and Ordinary Dividends | If you have >$1,500 in interest/dividends |
| C | Profit or Loss From Business | If you have business income |
| D | Capital Gains and Losses | If you sold assets |
| E | Supplemental Income and Loss | If you have rental/partnership income |
| F | Profit or Loss From Farming | If you have farm income |
| SE | Self-Employment Tax | If you're self-employed |

## Output Formats

### JSON with Bounding Boxes
```json
{
  "form_type": "IRS Form 1040",
  "page": 1,
  "fields": [
    {
      "id": "field_001",
      "type": "text_input",
      "label": "First name",
      "bbox_2d": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ]
}
```

### COCO Format
For object detection training:
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

## Data Location

Extracted forms are stored at:
```
/home/sweeden/projects/docling_data/tax_packet/docling_extracted/
├── f1040_docling.json
├── fw4_docling.json
├── fw9_docling.json
└── form1065_docling.json
```

## Next Steps

- [ ] Add Form 1040 Schedules (A, B, C, D, E, F, SE)
- [ ] Create ground truth annotations
- [ ] Train VLM on IRS forms
- [ ] Validate field detection accuracy
