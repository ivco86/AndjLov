"""
EXIF extraction utilities for AI Gallery
Extracts metadata from images using PIL/Pillow
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import fractions


def extract_exif_data(image_path: str) -> Optional[Dict]:
    """
    Extract EXIF data from an image file

    Returns a dict with keys matching exif_data table schema:
    - camera_make, camera_model, lens
    - iso, aperture, shutter_speed, focal_length
    - capture_date, gps_lat, gps_lon
    - orientation, flash, white_balance, metering_mode, exposure_mode, software
    """
    try:
        img = Image.open(image_path)
        exif_raw = img.getexif()

        if not exif_raw:
            return None

        # Build a cleaner EXIF dict with tag names
        exif_dict = {}
        for tag_id, value in exif_raw.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_dict[tag_name] = value

        # Extract GPS data if present
        gps_lat, gps_lon = None, None
        if 'GPSInfo' in exif_dict:
            gps_info = exif_dict['GPSInfo']
            gps_lat, gps_lon = _parse_gps(gps_info)

        # Parse capture date
        capture_date = None
        if 'DateTimeOriginal' in exif_dict:
            try:
                capture_date = datetime.strptime(
                    exif_dict['DateTimeOriginal'],
                    '%Y:%m:%d %H:%M:%S'
                ).isoformat()
            except:
                pass

        # Build result matching database schema
        result = {
            'camera_make': exif_dict.get('Make'),
            'camera_model': exif_dict.get('Model'),
            'lens': exif_dict.get('LensModel'),
            'iso': exif_dict.get('ISOSpeedRatings'),
            'aperture': _parse_aperture(exif_dict.get('FNumber')),
            'shutter_speed': _parse_shutter_speed(exif_dict.get('ExposureTime')),
            'focal_length': _parse_focal_length(exif_dict.get('FocalLength')),
            'capture_date': capture_date,
            'gps_lat': gps_lat,
            'gps_lon': gps_lon,
            'orientation': exif_dict.get('Orientation'),
            'flash': _parse_flash(exif_dict.get('Flash')),
            'white_balance': _parse_white_balance(exif_dict.get('WhiteBalance')),
            'metering_mode': _parse_metering_mode(exif_dict.get('MeteringMode')),
            'exposure_mode': _parse_exposure_mode(exif_dict.get('ExposureMode')),
            'software': exif_dict.get('Software')
        }

        return result

    except Exception as e:
        print(f"Error extracting EXIF from {image_path}: {e}")
        return None


def _parse_gps(gps_info) -> Tuple[Optional[float], Optional[float]]:
    """Parse GPS coordinates from EXIF GPS data"""
    try:
        gps_dict = {}
        for key, val in gps_info.items():
            tag_name = GPSTAGS.get(key, key)
            gps_dict[tag_name] = val

        if 'GPSLatitude' not in gps_dict or 'GPSLongitude' not in gps_dict:
            return None, None

        # Convert to decimal degrees
        lat = _convert_to_degrees(gps_dict['GPSLatitude'])
        lon = _convert_to_degrees(gps_dict['GPSLongitude'])

        # Apply direction (N/S, E/W)
        if gps_dict.get('GPSLatitudeRef') == 'S':
            lat = -lat
        if gps_dict.get('GPSLongitudeRef') == 'W':
            lon = -lon

        return lat, lon

    except Exception as e:
        print(f"Error parsing GPS: {e}")
        return None, None


def _convert_to_degrees(value):
    """Convert GPS coordinates to degrees"""
    d, m, s = value
    return float(d) + float(m) / 60 + float(s) / 3600


def _parse_aperture(value) -> Optional[float]:
    """Parse aperture f-number"""
    if value is None:
        return None

    try:
        if isinstance(value, tuple):
            return float(value[0]) / float(value[1])
        return float(value)
    except:
        return None


def _parse_shutter_speed(value) -> Optional[str]:
    """Parse shutter speed to readable format (e.g., '1/250')"""
    if value is None:
        return None

    try:
        if isinstance(value, tuple):
            num, denom = value
            if num == 1:
                return f"1/{denom}"
            else:
                # Convert to fraction
                frac = fractions.Fraction(num, denom).limit_denominator(8000)
                if frac.numerator == 1:
                    return f"1/{frac.denominator}"
                else:
                    return f"{frac.numerator}/{frac.denominator}"
        return str(value)
    except:
        return None


def _parse_focal_length(value) -> Optional[float]:
    """Parse focal length in mm"""
    if value is None:
        return None

    try:
        if isinstance(value, tuple):
            return float(value[0]) / float(value[1])
        return float(value)
    except:
        return None


def _parse_flash(value) -> Optional[str]:
    """Parse flash mode"""
    if value is None:
        return None

    flash_modes = {
        0: 'No Flash',
        1: 'Fired',
        5: 'Fired, Return not detected',
        7: 'Fired, Return detected',
        9: 'On, Fired',
        13: 'On, Return not detected',
        15: 'On, Return detected',
        16: 'Off',
        24: 'Auto, Did not fire',
        25: 'Auto, Fired',
        29: 'Auto, Fired, Return not detected',
        31: 'Auto, Fired, Return detected',
        32: 'No flash function',
        65: 'Red-eye reduction',
        69: 'Red-eye reduction, Return not detected',
        71: 'Red-eye reduction, Return detected',
        73: 'Red-eye reduction, On, Fired',
        77: 'Red-eye reduction, On, Return not detected',
        79: 'Red-eye reduction, On, Return detected',
        89: 'Red-eye reduction, Auto, Fired',
        93: 'Red-eye reduction, Auto, Fired, Return not detected',
        95: 'Red-eye reduction, Auto, Fired, Return detected'
    }

    return flash_modes.get(value, f'Unknown ({value})')


def _parse_white_balance(value) -> Optional[str]:
    """Parse white balance mode"""
    if value is None:
        return None

    wb_modes = {
        0: 'Auto',
        1: 'Manual'
    }

    return wb_modes.get(value, f'Unknown ({value})')


def _parse_metering_mode(value) -> Optional[str]:
    """Parse metering mode"""
    if value is None:
        return None

    metering_modes = {
        0: 'Unknown',
        1: 'Average',
        2: 'Center-weighted average',
        3: 'Spot',
        4: 'Multi-spot',
        5: 'Pattern',
        6: 'Partial',
        255: 'Other'
    }

    return metering_modes.get(value, f'Unknown ({value})')


def _parse_exposure_mode(value) -> Optional[str]:
    """Parse exposure mode"""
    if value is None:
        return None

    exposure_modes = {
        0: 'Auto',
        1: 'Manual',
        2: 'Auto bracket'
    }

    return exposure_modes.get(value, f'Unknown ({value})')


def get_readable_exif_summary(exif_dict: Dict) -> str:
    """Generate a human-readable summary of EXIF data"""
    parts = []

    if exif_dict.get('camera_make') or exif_dict.get('camera_model'):
        camera = f"{exif_dict.get('camera_make', '')} {exif_dict.get('camera_model', '')}".strip()
        parts.append(f"📷 {camera}")

    if exif_dict.get('lens'):
        parts.append(f"🔍 {exif_dict['lens']}")

    settings = []
    if exif_dict.get('focal_length'):
        settings.append(f"{exif_dict['focal_length']:.0f}mm")
    if exif_dict.get('aperture'):
        settings.append(f"f/{exif_dict['aperture']:.1f}")
    if exif_dict.get('shutter_speed'):
        settings.append(f"{exif_dict['shutter_speed']}s")
    if exif_dict.get('iso'):
        settings.append(f"ISO {exif_dict['iso']}")

    if settings:
        parts.append("⚙️ " + " • ".join(settings))

    if exif_dict.get('capture_date'):
        try:
            dt = datetime.fromisoformat(exif_dict['capture_date'])
            parts.append(f"📅 {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            pass

    if exif_dict.get('gps_lat') and exif_dict.get('gps_lon'):
        parts.append(f"📍 {exif_dict['gps_lat']:.6f}, {exif_dict['gps_lon']:.6f}")

    return '\n'.join(parts) if parts else 'No EXIF data available'
