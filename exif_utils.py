"""
EXIF Metadata Extraction Utilities
Extracts comprehensive camera metadata from images
"""

import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime


def extract_exif_data(image_path):
    """
    Extract all EXIF metadata from an image file

    Args:
        image_path: Path to the image file

    Returns:
        dict: Comprehensive EXIF data or None if not available
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if not exif_data:
            return None

        # Parse EXIF data
        parsed_exif = {}

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            parsed_exif[tag_name] = value

        # Extract structured data
        result = {
            'camera_make': parsed_exif.get('Make', '').strip(),
            'camera_model': parsed_exif.get('Model', '').strip(),
            'lens_model': parsed_exif.get('LensModel', '').strip(),
            'iso': _parse_int(parsed_exif.get('ISOSpeedRatings')),
            'aperture': _parse_aperture(parsed_exif.get('FNumber')),
            'shutter_speed': _parse_shutter_speed(parsed_exif.get('ExposureTime')),
            'focal_length': _parse_focal_length(parsed_exif.get('FocalLength')),
            'flash': _parse_flash(parsed_exif.get('Flash')),
            'white_balance': parsed_exif.get('WhiteBalance'),
            'metering_mode': parsed_exif.get('MeteringMode'),
            'exposure_mode': parsed_exif.get('ExposureMode'),
            'exposure_compensation': _parse_exposure_comp(parsed_exif.get('ExposureCompensation')),
            'orientation': parsed_exif.get('Orientation'),
            'date_taken': _parse_datetime(parsed_exif.get('DateTimeOriginal') or parsed_exif.get('DateTime')),
            'gps_latitude': None,
            'gps_longitude': None
        }

        # Extract GPS coordinates if available
        gps_data = parsed_exif.get('GPSInfo')
        if gps_data:
            coords = _parse_gps_coordinates(gps_data)
            if coords:
                result['gps_latitude'] = coords['latitude']
                result['gps_longitude'] = coords['longitude']

        # Clean up None values to empty strings for database storage
        for key in result:
            if result[key] is None:
                result[key] = '' if isinstance(result[key], str) or key in ['camera_make', 'camera_model', 'lens_model'] else None

        return result

    except Exception as e:
        print(f"Error extracting EXIF from {image_path}: {e}")
        return None


def _parse_int(value):
    """Parse integer value safely"""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            return int(value[0])
        return int(value)
    except:
        return None


def _parse_aperture(value):
    """Parse aperture f-number (e.g., 2.8)"""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return round(value[0] / value[1], 1)
        return round(float(value), 1)
    except:
        return None


def _parse_shutter_speed(value):
    """Parse shutter speed to decimal seconds"""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return value[0] / value[1]
        return float(value)
    except:
        return None


def _parse_focal_length(value):
    """Parse focal length in mm"""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return round(value[0] / value[1], 1)
        return round(float(value), 1)
    except:
        return None


def _parse_flash(value):
    """Parse flash status (0=no flash, 1=flash fired)"""
    if value is None:
        return None
    try:
        # Flash is a bitmask, check if bit 0 is set (flash fired)
        return 1 if (int(value) & 0x01) else 0
    except:
        return None


def _parse_exposure_comp(value):
    """Parse exposure compensation in EV"""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return round(value[0] / value[1], 2)
        return round(float(value), 2)
    except:
        return None


def _parse_datetime(value):
    """Parse EXIF datetime to ISO format"""
    if not value:
        return None
    try:
        # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
        dt = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
        return dt.isoformat()
    except:
        return None


def _parse_gps_coordinates(gps_info):
    """
    Parse GPS coordinates from EXIF GPSInfo

    Args:
        gps_info: GPS metadata dictionary

    Returns:
        dict: {'latitude': float, 'longitude': float} or None
    """
    try:
        # Decode GPSInfo tags
        gps_data = {}
        for key, value in gps_info.items():
            tag_name = GPSTAGS.get(key, key)
            gps_data[tag_name] = value

        # Extract latitude
        lat = gps_data.get('GPSLatitude')
        lat_ref = gps_data.get('GPSLatitudeRef')

        # Extract longitude
        lon = gps_data.get('GPSLongitude')
        lon_ref = gps_data.get('GPSLongitudeRef')

        if not all([lat, lat_ref, lon, lon_ref]):
            return None

        # Convert to decimal degrees
        latitude = _convert_to_degrees(lat)
        if lat_ref == 'S':
            latitude = -latitude

        longitude = _convert_to_degrees(lon)
        if lon_ref == 'W':
            longitude = -longitude

        return {
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6)
        }

    except Exception as e:
        print(f"Error parsing GPS coordinates: {e}")
        return None


def _convert_to_degrees(gps_coord):
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal degrees

    Args:
        gps_coord: Tuple of (degrees, minutes, seconds) as rationals

    Returns:
        float: Decimal degrees
    """
    d = gps_coord[0][0] / gps_coord[0][1]
    m = gps_coord[1][0] / gps_coord[1][1]
    s = gps_coord[2][0] / gps_coord[2][1]

    return d + (m / 60.0) + (s / 3600.0)


def get_camera_list_from_exif_data(exif_records):
    """
    Extract unique camera makes/models from a list of EXIF records

    Args:
        exif_records: List of EXIF data dictionaries

    Returns:
        list: Unique camera combinations [{"make": "Canon", "model": "EOS 5D Mark IV", "count": 42}, ...]
    """
    cameras = {}

    for record in exif_records:
        make = record.get('camera_make', '').strip()
        model = record.get('camera_model', '').strip()

        if make or model:
            key = f"{make}|{model}"
            if key in cameras:
                cameras[key]['count'] += 1
            else:
                cameras[key] = {
                    'make': make,
                    'model': model,
                    'count': 1
                }

    # Convert to list and sort by count
    camera_list = list(cameras.values())
    camera_list.sort(key=lambda x: x['count'], reverse=True)

    return camera_list


def format_exif_for_display(exif_data):
    """
    Format EXIF data for human-readable display

    Args:
        exif_data: EXIF data dictionary

    Returns:
        dict: Formatted display strings
    """
    if not exif_data:
        return {}

    result = {}

    # Camera
    if exif_data.get('camera_make') or exif_data.get('camera_model'):
        camera = f"{exif_data.get('camera_make', '')} {exif_data.get('camera_model', '')}".strip()
        result['camera'] = camera

    # Lens
    if exif_data.get('lens_model'):
        result['lens'] = exif_data['lens_model']

    # Settings
    settings_parts = []

    if exif_data.get('focal_length'):
        settings_parts.append(f"{exif_data['focal_length']}mm")

    if exif_data.get('aperture'):
        settings_parts.append(f"f/{exif_data['aperture']}")

    if exif_data.get('shutter_speed'):
        speed = exif_data['shutter_speed']
        if speed < 1:
            settings_parts.append(f"1/{int(1/speed)}s")
        else:
            settings_parts.append(f"{speed}s")

    if exif_data.get('iso'):
        settings_parts.append(f"ISO {exif_data['iso']}")

    if settings_parts:
        result['settings'] = ' · '.join(settings_parts)

    # Exposure compensation
    if exif_data.get('exposure_compensation'):
        ev = exif_data['exposure_compensation']
        if ev > 0:
            result['exposure_compensation'] = f"+{ev} EV"
        elif ev < 0:
            result['exposure_compensation'] = f"{ev} EV"

    # Flash
    if exif_data.get('flash') == 1:
        result['flash'] = "Flash fired"
    elif exif_data.get('flash') == 0:
        result['flash'] = "No flash"

    # Date taken
    if exif_data.get('date_taken'):
        try:
            dt = datetime.fromisoformat(exif_data['date_taken'])
            result['date_taken'] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            result['date_taken'] = exif_data['date_taken']

    # GPS
    if exif_data.get('gps_latitude') and exif_data.get('gps_longitude'):
        lat = exif_data['gps_latitude']
        lon = exif_data['gps_longitude']
        result['gps'] = f"{lat}, {lon}"
        result['gps_maps_url'] = f"https://www.google.com/maps?q={lat},{lon}"

    return result
