# Export Features Documentation

This document describes the PDF catalog and metadata export features for AI Gallery.

## Overview

The AI Gallery now supports exporting image metadata and generating professional PDF catalogs with the following features:

- **CSV Export**: Export image metadata to CSV format for use in spreadsheets
- **JSON Export**: Export image metadata to JSON format with optional summary statistics
- **PDF Catalog**: Generate professional PDF catalogs with images, descriptions, and metadata

## API Endpoints

### 1. Export Selected Images to CSV

**Endpoint**: `POST /api/export/images/csv`

**Description**: Export metadata for selected images to CSV format

**Request Body**:
```json
{
  "image_ids": [1, 2, 3, 4]
}
```

**Response**: CSV file download with filename `image_metadata.csv`

**Example using curl**:
```bash
curl -X POST http://localhost:5000/api/export/images/csv \
  -H "Content-Type: application/json" \
  -d '{"image_ids": [1, 2, 3]}' \
  --output metadata.csv
```

---

### 2. Export Selected Images to JSON

**Endpoint**: `POST /api/export/images/json`

**Description**: Export metadata for selected images to JSON format with optional summary

**Request Body**:
```json
{
  "image_ids": [1, 2, 3, 4],
  "include_summary": true
}
```

**Parameters**:
- `image_ids` (required): Array of image IDs to export
- `include_summary` (optional): Include summary statistics (default: true)

**Response**: JSON file download with filename `image_metadata.json`

**Example using curl**:
```bash
curl -X POST http://localhost:5000/api/export/images/json \
  -H "Content-Type: application/json" \
  -d '{"image_ids": [1, 2, 3], "include_summary": true}' \
  --output metadata.json
```

**Sample JSON Output**:
```json
{
  "summary": {
    "total_images": 3,
    "analyzed_images": 3,
    "favorite_images": 1,
    "total_size_mb": 12.45,
    "unique_tags": 15,
    "exported_at": "2025-11-18T10:30:00"
  },
  "images": [
    {
      "id": 1,
      "filename": "photo.jpg",
      "description": "A beautiful sunset over the ocean",
      "tags": "sunset, ocean, nature",
      "width": 1920,
      "height": 1080,
      "file_size": "2.34 MB"
    }
  ]
}
```

---

### 3. Export Board Images to CSV

**Endpoint**: `GET /api/export/boards/<board_id>/csv`

**Description**: Export all images from a specific board to CSV format

**Parameters**:
- `board_id` (in URL): The board ID

**Response**: CSV file download with filename `{board_name}_metadata.csv`

**Example using curl**:
```bash
curl http://localhost:5000/api/export/boards/1/csv --output board_metadata.csv
```

---

### 4. Export Board Images to JSON

**Endpoint**: `GET /api/export/boards/<board_id>/json`

**Description**: Export all images from a specific board to JSON format with summary

**Parameters**:
- `board_id` (in URL): The board ID

**Response**: JSON file download with filename `{board_name}_metadata.json`

**Example using curl**:
```bash
curl http://localhost:5000/api/export/boards/1/json --output board_metadata.json
```

---

### 5. Generate PDF Catalog for Board

**Endpoint**: `POST /api/export/boards/<board_id>/pdf`

**Description**: Generate a professional PDF catalog for all images in a board

**Parameters**:
- `board_id` (in URL): The board ID

**Request Body** (optional):
```json
{
  "page_size": "A4",
  "orientation": "portrait"
}
```

**Options**:
- `page_size`: "A4" or "letter" (default: "A4")
- `orientation`: "portrait" or "landscape" (default: "portrait")

**Response**: PDF file download with filename `{board_name}_catalog.pdf`

**Example using curl**:
```bash
curl -X POST http://localhost:5000/api/export/boards/1/pdf \
  -H "Content-Type: application/json" \
  -d '{"page_size": "A4", "orientation": "portrait"}' \
  --output catalog.pdf
```

---

### 6. Generate PDF Catalog from Selected Images

**Endpoint**: `POST /api/export/images/pdf`

**Description**: Generate a professional PDF catalog from selected images

**Request Body**:
```json
{
  "image_ids": [1, 2, 3, 4],
  "title": "My Photo Collection",
  "subtitle": "Selected images from 2025",
  "page_size": "A4",
  "orientation": "portrait"
}
```

**Parameters**:
- `image_ids` (required): Array of image IDs to include
- `title` (optional): Catalog title (default: "Image Catalog")
- `subtitle` (optional): Catalog subtitle (default: auto-generated with date and count)
- `page_size` (optional): "A4" or "letter" (default: "A4")
- `orientation` (optional): "portrait" or "landscape" (default: "portrait")

**Response**: PDF file download with filename `{title}.pdf`

**Example using curl**:
```bash
curl -X POST http://localhost:5000/api/export/images/pdf \
  -H "Content-Type: application/json" \
  -d '{
    "image_ids": [1, 2, 3],
    "title": "My Favorites",
    "page_size": "letter",
    "orientation": "landscape"
  }' \
  --output my_catalog.pdf
```

---

## CSV Export Format

The CSV export includes the following fields:

| Field | Description |
|-------|-------------|
| id | Image ID |
| filename | Image filename |
| filepath | Path to image file |
| description | AI-generated description |
| tags | Comma-separated tags |
| width | Image width in pixels |
| height | Image height in pixels |
| file_size | File size (formatted as MB/KB) |
| media_type | Type: "image" or "video" |
| is_favorite | "Yes" or "No" |
| created_at | Date added to gallery |
| updated_at | Last update date |
| analyzed_at | AI analysis date |

## PDF Catalog Features

The PDF catalogs include:

- **Professional Layout**: Clean, modern design with proper spacing
- **Image Thumbnails**: Optimized image display (uses thumbnails when available)
- **Rich Metadata**:
  - Filename
  - AI-generated description
  - Image dimensions
  - File size
  - Tags
  - Creation date
- **Customizable**: Support for different page sizes and orientations
- **Board Context**: Board name and description as catalog title/subtitle

## Usage Examples

### Example 1: Export Favorite Images to JSON

```python
import requests

# Get all favorite images
response = requests.get('http://localhost:5000/api/images?favorite=true')
images = response.json()['images']

# Extract image IDs
image_ids = [img['id'] for img in images]

# Export to JSON
export_response = requests.post(
    'http://localhost:5000/api/export/images/json',
    json={'image_ids': image_ids, 'include_summary': True}
)

with open('favorites.json', 'wb') as f:
    f.write(export_response.content)
```

### Example 2: Generate PDF Catalog for a Board

```python
import requests

board_id = 1

# Generate PDF catalog
response = requests.post(
    f'http://localhost:5000/api/export/boards/{board_id}/pdf',
    json={'page_size': 'A4', 'orientation': 'portrait'}
)

with open('board_catalog.pdf', 'wb') as f:
    f.write(response.content)
```

### Example 3: Batch Export All Boards

```python
import requests

# Get all boards
boards_response = requests.get('http://localhost:5000/api/boards')
boards = boards_response.json()['boards']

# Export each board to CSV and PDF
for board in boards:
    board_id = board['id']
    board_name = board['name'].replace(' ', '_')

    # Export to CSV
    csv_response = requests.get(f'http://localhost:5000/api/export/boards/{board_id}/csv')
    with open(f'{board_name}.csv', 'wb') as f:
        f.write(csv_response.content)

    # Export to PDF
    pdf_response = requests.post(f'http://localhost:5000/api/export/boards/{board_id}/pdf')
    with open(f'{board_name}.pdf', 'wb') as f:
        f.write(pdf_response.content)
```

## Installation

The export features require the `reportlab` library for PDF generation. Install it with:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install reportlab==4.0.7
```

## Technical Details

### PDF Generation

- **Library**: ReportLab
- **Page Sizes**: A4 (210 x 297 mm), Letter (8.5 x 11 inches)
- **Orientation**: Portrait or Landscape
- **Images Per Page**: ~3 images with metadata
- **Image Quality**: Uses thumbnails when available for optimal file size

### CSV Export

- **Format**: Standard CSV with header row
- **Encoding**: UTF-8
- **Delimiter**: Comma
- **Quoting**: Automatic for fields containing special characters

### JSON Export

- **Format**: Standard JSON
- **Encoding**: UTF-8
- **Pretty Print**: Enabled (2-space indentation)
- **Summary Stats**: Included by default

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful export
- `400 Bad Request`: Missing or invalid parameters
- `404 Not Found`: Board or images not found
- `500 Internal Server Error`: Export generation failed

Error responses include a JSON message:

```json
{
  "error": "Description of the error"
}
```

## Performance Considerations

- **PDF Generation**: May take several seconds for large boards (50+ images)
- **Thumbnails**: Using thumbnails significantly speeds up PDF generation
- **Memory**: Large exports may require substantial memory
- **File Cleanup**: Temporary PDF files are automatically cleaned up after download

## Future Enhancements

Potential features for future versions:

- Custom PDF templates and branding
- Multiple images per row in PDF layouts
- Excel (.xlsx) export format
- Batch export of multiple boards
- Scheduled/automated exports
- Custom field selection for CSV/JSON exports
- PDF password protection
- Watermarking support

## Support

For issues or questions about the export features, please refer to the main README.md or create an issue in the project repository.
