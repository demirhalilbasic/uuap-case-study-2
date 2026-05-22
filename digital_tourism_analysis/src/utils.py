import os
CITY_COLORS = {
    'Athens': '#FF6B35',
    'Malaga': '#004E89',
    'Istanbul': '#1A936F',
    'Naples': '#C84B31',
    'Rome': '#8B2FC9',
    'Valencia': '#E8A838'
}
CITY_CENTERS = {
    'Athens': (37.9838, 23.7275),
    'Malaga': (36.7213, -4.4213),
    'Istanbul': (41.0082, 28.9784),
    'Naples': (40.8518, 14.2681),
    'Rome': (41.9028, 12.4964),
    'Valencia': (39.4699, -0.3774)
}
CITY_CURRENCIES = {
    'Athens': 'EUR',
    'Malaga': 'EUR',
    'Istanbul': 'TRY',
    'Naples': 'EUR',
    'Rome': 'EUR',
    'Valencia': 'EUR'
}
CITY_FILES = {
    'Athens': 'athens',
    'Malaga': 'malaga',
    'Istanbul': 'istanbul',
    'Naples': 'naples',
    'Rome': 'rome',
    'Valencia': 'valencia'
}
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
