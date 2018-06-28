import os

project = 'bubo-1'
zone = 'us-west1-b'
gs_bucket = 'bubo-data'

# TODO Find a cleaner way to do this
_maybe_code_dirs = [
    '/Users/danb/hack/bubo/features',  # Local dev (osx)
    '/app',  # Remote deploy (docker/linux)
]
for _maybe_code_dir in _maybe_code_dirs:
    if os.path.exists(_maybe_code_dir):
        code_dir = _maybe_code_dir
        break
else:
    raise Exception(f'Found none of these dirs to use as code_dir: {_maybe_code_dirs}')

bin_dir = f'{code_dir}/bin'
data_dir = f'{code_dir}/data'
cache_dir = f'{data_dir}/cache'

standard_sample_rate_hz = 22050  # Can resolve 11025Hz (by Nyquist), which most/all birds are below
default_log_ylim_min_hz = 512  # Most/all birds are above 512Hz (but make sure to clip noise below 512Hz)

unk_species = '_UNK'
unk_species_com_name = 'Unknown'
unk_species_taxon_id = 'TC___UNK'
unk_species_species_code = '___unk'

mul_species = '_MUL'
mul_species_com_name = 'Multiple species'
mul_species_taxon_id = 'TC___MUL'
mul_species_species_code = '___mul'

no_species = '_NON'
no_species_com_name = 'No species'
no_species_taxon_id = 'TC___NON'
no_species_species_code = '___non'

##
# countries

countries_na = ['United States', 'Canada', 'Mexico']

##
# com_names

# The first set of hand-curated species I started training with, based on Mitchell Canyon / Coyote Hills checklists
#   - XXX Not very valuable to hold on to
com_names_dan_168 = [
    "Mountain Quail", "Scaled Quail", "California Quail", "Montezuma Quail", "Sooty Grouse", "Wild Turkey",
    "Double-crested Cormorant", "Least Bittern", "Great Blue Heron", "Great Egret", "Snowy Egret", "Green Heron",
    "Black-crowned Night-Heron", "Turkey Vulture", "California Condor", "Osprey", "White-tailed Kite", "Golden Eagle",
    "Northern Harrier", "Sharp-shinned Hawk", "Cooper's Hawk", "Bald Eagle", "Red-shouldered Hawk", "Red-tailed Hawk",
    "Killdeer", "Rock Pigeon", "Band-tailed Pigeon", "Eurasian Collared-Dove", "Mourning Dove", "Western Screech-Owl",
    "Great Horned Owl", "Burrowing Owl", "Common Nighthawk", "Common Poorwill", "White-throated Swift",
    "Anna's Hummingbird", "Rufous Hummingbird", "Allen's Hummingbird", "Belted Kingfisher", "Acorn Woodpecker",
    "Red-breasted Sapsucker", "Nuttall's Woodpecker", "Downy Woodpecker", "Hairy Woodpecker", "Northern Flicker",
    "Pileated Woodpecker", "American Kestrel", "Merlin", "Peregrine Falcon", "Olive-sided Flycatcher",
    "Western Wood-Pewee", "Pacific-slope Flycatcher", "Black Phoebe", "Say's Phoebe", "Ash-throated Flycatcher",
    "Western Kingbird", "Hutton's Vireo", "Cassin's Vireo", "Warbling Vireo", "Steller's Jay", "California Scrub-Jay",
    "Black-billed Magpie", "Yellow-billed Magpie", "Clark's Nutcracker", "American Crow", "Common Raven",
    "Northern Rough-winged Swallow", "Purple Martin", "Tree Swallow", "Violet-green Swallow", "Barn Swallow",
    "Cliff Swallow", "Carolina Chickadee", "Black-capped Chickadee", "Mountain Chickadee", "Chestnut-backed Chickadee",
    "Boreal Chickadee", "Oak Titmouse", "Tufted Titmouse", "Black-crested Titmouse", "Bushtit", "Red-breasted Nuthatch",
    "White-breasted Nuthatch", "Pygmy Nuthatch", "Brown Creeper", "Rock Wren", "Canyon Wren", "House Wren",
    "Pacific Wren", "Winter Wren", "Sedge Wren", "Marsh Wren", "Carolina Wren", "Bewick's Wren", "Cactus Wren",
    "Blue-gray Gnatcatcher", "Golden-crowned Kinglet", "Ruby-crowned Kinglet", "Wrentit", "Western Bluebird",
    "Mountain Bluebird", "Swainson's Thrush", "Hermit Thrush", "Wood Thrush", "American Robin", "Varied Thrush",
    "California Thrasher", "Northern Mockingbird", "European Starling", "Cedar Waxwing", "Phainopepla",
    "Black-and-white Warbler", "Orange-crowned Warbler", "Nashville Warbler", "MacGillivray's Warbler",
    "Common Yellowthroat", "American Redstart", "Cape May Warbler", "Cerulean Warbler", "Magnolia Warbler",
    "Bay-breasted Warbler", "Yellow Warbler", "Black-throated Blue Warbler", "Yellow-rumped Warbler",
    "Yellow-throated Warbler", "Black-throated Gray Warbler", "Townsend's Warbler", "Hermit Warbler",
    "Golden-cheeked Warbler", "Black-throated Green Warbler", "Wilson's Warbler", "Chipping Sparrow", "Lark Sparrow",
    "Fox Sparrow", "Dark-eyed Junco", "White-crowned Sparrow", "Golden-crowned Sparrow", "Bell's Sparrow",
    "Savannah Sparrow", "Song Sparrow", "Lincoln's Sparrow", "Swamp Sparrow", "Canyon Towhee", "California Towhee",
    "Rufous-crowned Sparrow", "Spotted Towhee", "Summer Tanager", "Scarlet Tanager", "Western Tanager",
    "Northern Cardinal", "Black-headed Grosbeak", "Blue Grosbeak", "Lazuli Bunting", "Western Meadowlark",
    "Hooded Oriole", "Bullock's Oriole", "Red-winged Blackbird", "Tricolored Blackbird", "Brown-headed Cowbird",
    "Brewer's Blackbird", "Common Grackle", "Great-tailed Grackle", "Black Rosy-Finch", "House Finch", "Purple Finch",
    "Pine Siskin", "Lesser Goldfinch", "American Goldfinch", "House Sparrow",
]

# All xc com_name's that have ≥1 rec in CA
#   from notebooks import *
#   import pprint
#   (xc.metadata
#       [lambda df: pd.notnull(df.species)]
#       [lambda df: (df.country == 'United States') & df.locality.str.endswith('California')]
#       .pipe(df_remove_unused_categories).com_name.sort_values()
#       .pipe(lambda s:
#           pprint.pprint(s.unique().tolist(), indent=4, width=120, compact=True)
#       )
#   )
com_names_ca = [
    'Snow Goose', 'Greater White-fronted Goose', 'Brant Goose', 'Cackling Goose', 'Canada Goose', 'Tundra Swan',
    'Egyptian Goose', 'Cinnamon Teal', 'Gadwall', 'American Wigeon', 'Mallard', 'Green-winged Teal', 'Redhead',
    'Surf Scoter', 'Bufflehead', 'Mountain Quail', 'California Quail', "Gambel's Quail", 'Sooty Grouse', 'Wild Turkey',
    'Red-throated Loon', 'Great Northern Loon', 'Pied-billed Grebe', 'Horned Grebe', 'Red-necked Grebe',
    'Western Grebe', "Clark's Grebe", 'Black-footed Albatross', 'Sooty Shearwater', 'Black-vented Shearwater',
    "Brandt's Cormorant", 'Pelagic Cormorant', 'Double-crested Cormorant', 'American Bittern', 'Least Bittern',
    'Great Blue Heron', 'Great Egret', 'Snowy Egret', 'Green Heron', 'Black-crowned Night Heron', 'White-faced Ibis',
    'Western Osprey', 'White-tailed Kite', 'Northern Harrier', "Cooper's Hawk", 'Common Black Hawk',
    'Red-shouldered Hawk', "Swainson's Hawk", 'Red-tailed Hawk', 'Black Rail', "Ridgway's Rail", 'Virginia Rail',
    'Sora', 'Common Gallinule', 'American Coot', 'Sandhill Crane', 'Black-necked Stilt', 'American Avocet',
    'Black Oystercatcher', 'Grey Plover', 'Pacific Golden Plover', 'Snowy Plover', 'Semipalmated Plover', 'Killdeer',
    'Whimbrel', 'Long-billed Curlew', 'Marbled Godwit', 'Black Turnstone', 'Surfbird', 'Dunlin', 'Rock Sandpiper',
    'Least Sandpiper', 'Western Sandpiper', 'Short-billed Dowitcher', 'Long-billed Dowitcher', 'Red-necked Phalarope',
    'Spotted Sandpiper', 'Solitary Sandpiper', 'Wandering Tattler', 'Greater Yellowlegs', 'Willet', 'Common Murre',
    'Marbled Murrelet', "Bonaparte's Gull", 'Laughing Gull', "Heermann's Gull", 'Mew Gull', 'Ring-billed Gull',
    'Western Gull', 'Yellow-footed Gull', 'California Gull', 'Glaucous-winged Gull', 'Least Tern', 'Gull-billed Tern',
    'Caspian Tern', 'Black Tern', "Forster's Tern", 'Royal Tern', 'Elegant Tern', 'Black Skimmer', 'Rock Dove',
    'Band-tailed Pigeon', 'Eurasian Collared Dove', 'African Collared Dove', 'Inca Dove', 'Common Ground Dove',
    'White-winged Dove', 'Mourning Dove', 'Greater Roadrunner', 'Yellow-billed Cuckoo', 'Western Barn Owl',
    'Flammulated Owl', 'Western Screech Owl', 'Great Horned Owl', 'Elf Owl', 'Burrowing Owl', 'Spotted Owl',
    'Long-eared Owl', 'Northern Saw-whet Owl', 'Lesser Nighthawk', 'Common Nighthawk', 'Common Poorwill',
    'Eastern Whip-poor-will', 'Mexican Whip-poor-will', "Vaux's Swift", 'White-throated Swift',
    'Black-chinned Hummingbird', "Anna's Hummingbird", "Costa's Hummingbird", 'Broad-tailed Hummingbird',
    "Allen's Hummingbird", 'Calliope Hummingbird', 'Belted Kingfisher', 'Acorn Woodpecker', 'Gila Woodpecker',
    "Williamson's Sapsucker", 'Red-breasted Sapsucker', 'Ladder-backed Woodpecker', "Nuttall's Woodpecker",
    'Downy Woodpecker', 'Hairy Woodpecker', 'White-headed Woodpecker', 'Black-backed Woodpecker', 'Northern Flicker',
    'Pileated Woodpecker', 'American Kestrel', 'Merlin', 'Peregrine Falcon', 'Cockatiel', 'Rose-ringed Parakeet',
    'Yellow-chevroned Parakeet', 'Red-crowned Amazon', 'Lilac-crowned Amazon', 'Yellow-headed Amazon',
    'Nanday Parakeet', 'Red-masked Parakeet', 'Olive-sided Flycatcher', 'Western Wood Pewee', 'Willow Flycatcher',
    'Least Flycatcher', "Hammond's Flycatcher", 'American Grey Flycatcher', 'American Dusky Flycatcher',
    'Pacific-slope Flycatcher', 'Black Phoebe', 'Eastern Phoebe', "Say's Phoebe", 'Vermilion Flycatcher',
    'Dusky-capped Flycatcher', 'Ash-throated Flycatcher', 'Brown-crested Flycatcher', 'Tropical Kingbird',
    "Couch's Kingbird", "Cassin's Kingbird", 'Thick-billed Kingbird', 'Western Kingbird', 'Loggerhead Shrike',
    'White-eyed Vireo', "Bell's Vireo", 'Grey Vireo', "Hutton's Vireo", "Cassin's Vireo", 'Plumbeous Vireo',
    'Warbling Vireo', 'Red-eyed Vireo', 'Yellow-green Vireo', 'Grey Jay', 'Black-throated Magpie-Jay', 'Pinyon Jay',
    "Steller's Jay", 'Island Scrub Jay', 'California Scrub Jay', "Woodhouse's Scrub Jay", 'Yellow-billed Magpie',
    "Clark's Nutcracker", 'American Crow', 'Northern Raven', 'Horned Lark', 'Northern Rough-winged Swallow',
    'Purple Martin', 'Tree Swallow', 'Violet-green Swallow', 'Barn Swallow', 'American Cliff Swallow',
    'Mountain Chickadee', 'Chestnut-backed Chickadee', 'Oak Titmouse', 'Juniper Titmouse', 'Verdin', 'American Bushtit',
    'Red-breasted Nuthatch', 'White-breasted Nuthatch', 'Pygmy Nuthatch', 'Brown Creeper', 'Rock Wren', 'Canyon Wren',
    'House Wren', 'Pacific Wren', 'Winter Wren', 'Marsh Wren', "Bewick's Wren", 'Cactus Wren', 'Blue-grey Gnatcatcher',
    'California Gnatcatcher', 'Black-tailed Gnatcatcher', 'American Dipper', 'Red-whiskered Bulbul',
    'Golden-crowned Kinglet', 'Ruby-crowned Kinglet', 'Dusky Warbler', 'Wrentit', 'Japanese White-eye',
    'Western Bluebird', 'Mountain Bluebird', "Townsend's Solitaire", "Swainson's Thrush", 'Hermit Thrush',
    'Wood Thrush', 'American Robin', 'Varied Thrush', 'Grey Catbird', 'Curve-billed Thrasher', "Bendire's Thrasher",
    'California Thrasher', "Le Conte's Thrasher", 'Crissal Thrasher', 'Sage Thrasher', 'Northern Mockingbird',
    'Common Starling', 'Olive-backed Pipit', 'Red-throated Pipit', 'Buff-bellied Pipit', 'Cedar Waxwing', 'Phainopepla',
    'Lapland Longspur', 'Worm-eating Warbler', 'Northern Waterthrush', 'Orange-crowned Warbler', "Lucy's Warbler",
    'Nashville Warbler', "MacGillivray's Warbler", 'Mourning Warbler', 'Common Yellowthroat', 'Hooded Warbler',
    'American Redstart', 'Northern Parula', 'Magnolia Warbler', 'Bay-breasted Warbler', 'Chestnut-sided Warbler',
    'Pine Warbler', 'Myrtle Warbler', 'Prairie Warbler', "Grace's Warbler", 'Black-throated Grey Warbler',
    "Townsend's Warbler", 'Hermit Warbler', "Wilson's Warbler", 'Painted Whitestart', "Cassin's Sparrow",
    'Grasshopper Sparrow', 'Chipping Sparrow', 'Clay-colored Sparrow', 'Black-chinned Sparrow', "Brewer's Sparrow",
    'Black-throated Sparrow', 'Lark Sparrow', 'Fox Sparrow', 'Dark-eyed Junco', 'White-crowned Sparrow',
    'Golden-crowned Sparrow', 'White-throated Sparrow', 'Sagebrush Sparrow', "Bell's Sparrow", 'Vesper Sparrow',
    'Savannah Sparrow', 'Song Sparrow', "Lincoln's Sparrow", "Abert's Towhee", 'California Towhee',
    'Rufous-crowned Sparrow', 'Green-tailed Towhee', 'Spotted Towhee', 'Yellow-breasted Chat', 'Summer Tanager',
    'Western Tanager', 'Northern Cardinal', 'Rose-breasted Grosbeak', 'Black-headed Grosbeak', 'Blue Grosbeak',
    'Lazuli Bunting', 'Indigo Bunting', 'Yellow-headed Blackbird', 'Western Meadowlark', 'Hooded Oriole',
    "Bullock's Oriole", "Scott's Oriole", 'Red-winged Blackbird', 'Tricolored Blackbird', 'Brown-headed Cowbird',
    "Brewer's Blackbird", 'Great-tailed Grackle', 'Evening Grosbeak', 'Pine Grosbeak', 'House Finch', 'Purple Finch',
    "Cassin's Finch", 'Red Crossbill', 'Pine Siskin', 'Lesser Goldfinch', "Lawrence's Goldfinch", 'American Goldfinch',
    'House Sparrow', 'Scaly-breasted Munia', 'Pin-tailed Whydah',
]
