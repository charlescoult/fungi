
mush_attributes = [
    {
        ref: "cap.shape",
        values: [
            "bell",
            "conical",
            "convex",
            "flat",
            "knobbed",
            "sunken",
        ]
    },
    {
        ref: "cap.surface",
        values: [
            "fibrous",
            "groves",
            "scaly",
            "smooth",
        ]
    }
]

dss = [
    {
        'name': 'Mushroom Data Set (UCI)',
        'url': 'https://archive.ics.uci.edu/ml/datasets/mushroom',
        'mushroom-attributes': [
                "cap.shape",
                "cap.surface",
                "cap.color",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
        ],
    },
]


# 'notation' is the name used within the foreign dataset
# 'name' is the name used within my dataset
attributes_list = [
    {
        'name': 'edibility',
        'values': [
            {
                'notation': 'p',
                'name': 'poisonous',
            },
            {
                'notation': 'e',
                'name': 'edible',
            },
        ]
    },
    {
        'name': 'cap.shape',
        'values': [
            {
                'notation': 'x',
                'name': 'convex',
            },
            {
                'notation': 'b',
                'name': 'bell',
            },
            {
                'notation': 's',
                'name': 'sunken',
            },
            {
                'notation': 'f',
                'name': 'flat',
            },
            {
                'notation': 'k',
                'name': 'knobbed',
            },
            {
                'notation': 'c',
                'name': 'conical',
            },
        ]
    },
    {
        'name': 'cap.surface',
        'values': [
            {
                'notation': 's',
                'name': 'smooth',
            },
            {
                'notation': 'y',
                'name': 'scaly',
            },
            {
                'notation': 'f',
                'name': 'fibrous',
            },
            {
                'notation': 'g',
                'name': 'grooves',
            },
        ]
    },
    {
        'name': 'cap.color',
        'values': [
            {
                'notation': 'n',
                'name': 'brown',
            },
            {
                'notation': 'b',
                'name': 'buff',
            },
            {
                'notation': 'c',
                'name': 'cinnamon',
            },
            {
                'notation': 'g',
                'name': 'gray',
            },
            {
                'notation': 'r',
                'name': 'green',
            },
            {
                'notation': 'p',
                'name': 'pink',
            },
            {
                'notation': 'u',
                'name': 'purple',
            },
            {
                'notation': 'e',
                'name': 'red',
            },
            {
                'notation': 'w',
                'name': 'white',
            },
            {
                'notation': 'y',
                'name': 'yellow',
            },
        ]
    },
    {
        'name': 'bruises',
        'values': [
            {
                'notation': 't',
                'name': 'bruises',
            },
            {
                'notation': 'y',
                'name': 'no bruise',
            },
        ]
    },
    {
        'name': 'odor',
        'values': [
            {
                'notation': 'a',
                'name': 'almond',
            },
            {
                'notation': 'l',
                'name': 'anise',
            },
            {
                'notation': 'c',
                'name': 'creosote',
            },
            {
                'notation': 'y',
                'name': 'fishy',
            },
            {
                'notation': 'f',
                'name': 'foul',
            },
            {
                'notation': 'm',
                'name': 'musty',
            },
            {
                'notation': 'n',
                'name': 'none',
            },
            {
                'notation': 'p',
                'name': 'pungent',
            },
            {
                'notation': 's',
                'name': 'spicy',
            },
        ]
    },
    {
        'notation': 'gill.attachment',
        'name': 'gill.attachment',
        'values': [
            {
                'notation': 'a',
                'name': 'attached',
            },
            {
                'notation': 'd',
                'name': 'descending',
            },
            {
                'notation': 'f',
                'name': 'free',
            },
            {
                'notation': 'n',
                'name': 'notched',
            },
        ]
    },
    {
        'name': 'gill.spacing',
        'values': [
                'c',
            'w'
        ]
    },
    {
        'name': 'gill.size',
        'values': [
                'n',
            'b'
        ]
    },
    {
        'name': 'gill.color',
        'values': [
                'k',
            'n',
            'g',
            'p',
            'w',
            'h',
            'u',
            'e',
            'b',
            'r',
            'y',
            'o'
        ]
    },
    {
        'name': 'stalk.shape',
        'values': [
                'e',
            't'
        ]
    },
    {
        'name': 'stalk.root',
        'values': [
                'e',
            'c',
            'b',
            'r',
            '?'
        ]
    },
    {
        'name': 'stalk.surface.above-ring',
        'values': [
                's',
            'f',
            'k',
            'y'
        ]
    },
    {
        'name': 'stalk.surface.below-ring',
        'values': [
                's',
            'f',
            'y',
            'k'
        ]
    },
    {
        'name': 'stalk.color.above-ring',
        'values': [
                'w',
            'g',
            'p',
            'n',
            'b',
            'e',
            'o',
            'c',
            'y'
        ]
    },
    {
        'name': 'stalk.color.below-ring',
        'values': [
                'w',
            'p',
            'g',
            'b',
            'n',
            'e',
            'y',
            'o',
            'c'
        ]
    },
    {
        'name': 'veil.type',
        'values': [
                'p'
        ]
    },
    {
        'name': 'veil.color',
        'values': [
                'w',
            'n',
            'o',
            'y'
        ]
    },
    {
        'name': 'ring.number',
        'values': [
                'o',
            't',
            'n'
        ]
    },
    {
        'name': 'ring.type',
        'values': [
                'p',
            'e',
            'l',
            'f',
            'n'
        ]
    },
    {
        'name': 'spore-print.color',
        'values': [
                'k',
            'n',
            'u',
            'h',
            'w',
            'r',
            'o',
            'y',
            'b'
        ]
    },
    {
        'name': 'population',
        'values': [
                's',
            'n',
            'a',
            'v',
            'y',
            'c'
        ]
    },
    {
        'name': 'habitat',
        'values': [
                'u',
            'g',
            'm',
            'd',
            'p',
            'w',
            'l'
        ]
    }
]
