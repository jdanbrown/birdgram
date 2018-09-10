import os

project = 'bubo-1'
zone = 'us-west1-b'
gs_bucket = 'bubo-data'
gs_data_dir = f'gs://{gs_bucket}/v0/data'  # Mirror of data_dir in gs

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

# Dirs
#   - code/
bin_dir = f'{code_dir}/bin'
data_dir = f'{code_dir}/data'
#   - data/
cache_dir = f'{data_dir}/cache'
artifact_dir = f'{data_dir}/artifacts'
hand_labels_dir = f'{data_dir}/hand-labels'

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

countries = dict(
    na=['United States', 'Canada', 'Mexico'],
)

##
# com_names

com_names = dict(

    # WARNING Ensure this is the only n_species=4 dataset to avoid colliding with other n_species=4 model ids
    dan4=('ebird', ["Bewick's Wren", "House Wren", "Song Sparrow", "Spotted Towhee"]),
    dan5=('ebird', ["Bewick's Wren", "House Wren", "Song Sparrow", "Spotted Towhee", "Pacific-slope Flycatcher"]),

    ggow=('ebird', ["Great Gray Owl"]),

    # The first set of hand-curated species I started training with, based on Mitchell Canyon / Coyote Hills checklists
    #   - XXX Not very valuable to hold on to
    #   - WARNING Keep this ≠168 species to avoid colliding with n_species=168 model ids in the eval notebooks
    dan170=('ebird', [
        "Mountain Quail", "Scaled Quail", "California Quail", "Montezuma Quail", "Sooty Grouse", "Wild Turkey",
        "Double-crested Cormorant", "Least Bittern", "Great Blue Heron", "Great Egret", "Snowy Egret", "Green Heron",
        "Black-crowned Night-Heron", "Osprey", "White-tailed Kite", "Golden Eagle",
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
        "Gray Catbird", "California Thrasher", "Northern Mockingbird", "European Starling", "Cedar Waxwing", "Phainopepla",
        "Black-and-white Warbler", "Orange-crowned Warbler", "Nashville Warbler", "MacGillivray's Warbler",
        "Common Yellowthroat", "American Redstart", "Cape May Warbler", "Cerulean Warbler", "Magnolia Warbler",
        "Bay-breasted Warbler", "Yellow Warbler", "Blackpoll Warbler", "Black-throated Blue Warbler", "Palm Warbler",
        "Yellow-rumped Warbler",
        "Yellow-throated Warbler", "Black-throated Gray Warbler", "Townsend's Warbler", "Hermit Warbler",
        "Golden-cheeked Warbler", "Black-throated Green Warbler", "Wilson's Warbler", "Chipping Sparrow", "Lark Sparrow",
        "Fox Sparrow", "Dark-eyed Junco", "White-crowned Sparrow", "Golden-crowned Sparrow", "Bell's Sparrow",
        "Savannah Sparrow", "Song Sparrow", "Lincoln's Sparrow", "Swamp Sparrow", "Canyon Towhee", "California Towhee",
        "Rufous-crowned Sparrow", "Spotted Towhee", "Summer Tanager", "Scarlet Tanager", "Western Tanager",
        "Northern Cardinal", "Black-headed Grosbeak", "Blue Grosbeak", "Lazuli Bunting", "Western Meadowlark",
        "Hooded Oriole", "Bullock's Oriole", "Red-winged Blackbird", "Tricolored Blackbird", "Brown-headed Cowbird",
        "Brewer's Blackbird", "Common Grackle", "Great-tailed Grackle", "Black Rosy-Finch", "House Finch", "Purple Finch",
        "Pine Siskin", "Lesser Goldfinch", "American Goldfinch", "House Sparrow",
    ]),

    # All xc com_name's that have ≥1 rec in CA
    #   from notebooks import *
    #   import pprint
    #   (xc.metadata
    #       [lambda df: pd.notnull(df.species)]
    #       [lambda df: (df.country == 'United States') & df.locality.str.endswith('California')]
    #       .pipe(df_remove_unused_categories).com_name.sort_values()
    #       .pipe(puts, f=lambda s: len(s.unique()))
    #       .pipe(lambda s:
    #           pprint.pprint(s.unique().tolist(), indent=4, width=120, compact=True)
    #       )
    #   )
    ca=('xc', [
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
    ]),

    # All xc com_name's that have ≥1 rec in CA
    #   from notebooks import *
    #   import pprint
    #   (xc.metadata
    #       [lambda df: pd.notnull(df.species)]
    #       [lambda df: df.country == 'United States']
    #       .pipe(df_remove_unused_categories).com_name.sort_values()
    #       .pipe(puts, f=lambda s: len(s.unique()))
    #       .pipe(lambda s:
    #           pprint.pprint(s.unique().tolist(), indent=4, width=120, compact=True)
    #       )
    #   )
    us=('xc', [
        'Black-bellied Whistling Duck', 'Fulvous Whistling Duck', 'Emperor Goose', 'Snow Goose', "Ross's Goose",
        'Greater White-fronted Goose', 'Lesser White-fronted Goose', 'Tundra Bean Goose', 'Brant Goose', 'Cackling Goose',
        'Canada Goose', 'Nene', 'Mute Swan', 'Trumpeter Swan', 'Tundra Swan', 'Egyptian Goose', 'Wood Duck',
        'Blue-winged Teal', 'Cinnamon Teal', 'Northern Shoveler', 'Gadwall', 'Eurasian Wigeon', 'American Wigeon',
        'Laysan Duck', 'Hawaiian Duck', 'Mallard', 'American Black Duck', 'Mottled Duck', 'Northern Pintail',
        'Eurasian Teal', 'Green-winged Teal', 'Canvasback', 'Redhead', 'Ring-necked Duck', 'Greater Scaup', 'Lesser Scaup',
        "Steller's Eider", 'Spectacled Eider', 'King Eider', 'Common Eider', 'Harlequin Duck', 'Surf Scoter',
        'Black Scoter', 'Long-tailed Duck', 'Bufflehead', 'Common Goldeneye', "Barrow's Goldeneye", 'Hooded Merganser',
        'Common Merganser', 'Red-breasted Merganser', 'Ruddy Duck', 'Plain Chachalaca', 'Helmeted Guineafowl',
        'Mountain Quail', 'Northern Bobwhite', 'Scaled Quail', 'California Quail', "Gambel's Quail", 'Montezuma Quail',
        'Indian Peafowl', 'Chukar Partridge', 'Himalayan Snowcock', "Erckel's Francolin", 'Black Francolin',
        'Grey Francolin', 'Common Pheasant', 'Kalij Pheasant', 'Grey Partridge', 'Ruffed Grouse', 'Sage Grouse',
        'Gunnison Grouse', 'Spruce Grouse', 'Willow Ptarmigan', 'Rock Ptarmigan', 'White-tailed Ptarmigan', 'Dusky Grouse',
        'Sooty Grouse', 'Sharp-tailed Grouse', 'Greater Prairie Chicken', 'Lesser Prairie Chicken', 'Wild Turkey',
        'Red-throated Loon', 'Pacific Loon', 'Great Northern Loon', 'Yellow-billed Loon', 'Least Grebe',
        'Pied-billed Grebe', 'Horned Grebe', 'Red-necked Grebe', 'Black-necked Grebe', 'Western Grebe', "Clark's Grebe",
        'Laysan Albatross', 'Black-footed Albatross', 'Northern Fulmar', 'Hawaiian Petrel', 'Bonin Petrel',
        'Wedge-tailed Shearwater', 'Sooty Shearwater', "Newell's Shearwater", 'Black-vented Shearwater',
        'Red-tailed Tropicbird', 'Wood Stork', 'Great Frigatebird', 'Red-footed Booby', "Brandt's Cormorant",
        'Red-faced Cormorant', 'Pelagic Cormorant', 'Neotropic Cormorant', 'Double-crested Cormorant', 'Anhinga',
        'American White Pelican', 'Brown Pelican', 'American Bittern', 'Least Bittern', 'Great Blue Heron', 'Great Egret',
        'Snowy Egret', 'Little Blue Heron', 'Tricolored Heron', 'Reddish Egret', 'Western Cattle Egret', 'Green Heron',
        'Black-crowned Night Heron', 'Yellow-crowned Night Heron', 'American White Ibis', 'Glossy Ibis', 'White-faced Ibis',
        'Roseate Spoonbill', 'Black Vulture', 'Turkey Vulture', 'Western Osprey', 'White-tailed Kite',
        'Swallow-tailed Kite', 'Golden Eagle', 'Snail Kite', 'Mississippi Kite', 'Northern Harrier', 'Sharp-shinned Hawk',
        "Cooper's Hawk", 'Northern Goshawk', 'Bald Eagle', 'Common Black Hawk', "Harris's Hawk", 'White-tailed Hawk',
        'Grey Hawk', 'Red-shouldered Hawk', 'Broad-winged Hawk', 'Hawaiian Hawk', "Swainson's Hawk", 'Zone-tailed Hawk',
        'Red-tailed Hawk', 'Rough-legged Buzzard', 'Ferruginous Hawk', 'Yellow Rail', 'Black Rail', "Ridgway's Rail",
        'King Rail', 'Clapper Rail', 'Virginia Rail', 'Rufous-necked Wood Rail', 'Sora', 'Purple Gallinule',
        'Purple Swamphen', 'Common Gallinule', 'Hawaiian Coot', 'American Coot', 'Limpkin', 'Sandhill Crane',
        'Whooping Crane', 'Red-crowned Crane', 'Black-necked Stilt', 'American Avocet', 'American Oystercatcher',
        'Black Oystercatcher', 'Grey Plover', 'American Golden Plover', 'Pacific Golden Plover', 'Snowy Plover',
        "Wilson's Plover", 'Semipalmated Plover', 'Piping Plover', 'Killdeer', 'Mountain Plover', 'Northern Jacana',
        'Upland Sandpiper', 'Bristle-thighed Curlew', 'Whimbrel', 'Long-billed Curlew', 'Bar-tailed Godwit',
        'Hudsonian Godwit', 'Marbled Godwit', 'Ruddy Turnstone', 'Black Turnstone', 'Red Knot', 'Surfbird',
        'Sharp-tailed Sandpiper', 'Stilt Sandpiper', 'Red-necked Stint', 'Sanderling', 'Dunlin', 'Rock Sandpiper',
        'Purple Sandpiper', "Baird's Sandpiper", 'Least Sandpiper', 'White-rumped Sandpiper', 'Pectoral Sandpiper',
        'Semipalmated Sandpiper', 'Western Sandpiper', 'Short-billed Dowitcher', 'Long-billed Dowitcher',
        'American Woodcock', "Wilson's Snipe", 'Pin-tailed Snipe', "Wilson's Phalarope", 'Red-necked Phalarope',
        'Red Phalarope', 'Spotted Sandpiper', 'Solitary Sandpiper', 'Grey-tailed Tattler', 'Wandering Tattler',
        'Greater Yellowlegs', 'Willet', 'Lesser Yellowlegs', 'Wood Sandpiper', 'Pomarine Skua', 'Parasitic Jaeger',
        'Long-tailed Jaeger', 'Common Murre', 'Thick-billed Murre', 'Razorbill', 'Pigeon Guillemot', 'Marbled Murrelet',
        'Ancient Murrelet', 'Parakeet Auklet', 'Least Auklet', 'Whiskered Auklet', 'Crested Auklet', 'Horned Puffin',
        'Black-legged Kittiwake', 'Red-legged Kittiwake', "Sabine's Gull", "Bonaparte's Gull", 'Little Gull', "Ross's Gull",
        'Laughing Gull', "Franklin's Gull", "Heermann's Gull", 'Mew Gull', 'Ring-billed Gull', 'Western Gull',
        'Yellow-footed Gull', 'California Gull', 'Lesser Black-backed Gull', 'Slaty-backed Gull', 'Glaucous-winged Gull',
        'Glaucous Gull', 'Great Black-backed Gull', 'Brown Noddy', 'Black Noddy', 'White Tern', 'Sooty Tern',
        'Aleutian Tern', 'Least Tern', 'Gull-billed Tern', 'Caspian Tern', 'Black Tern', 'Roseate Tern', 'Common Tern',
        'Arctic Tern', "Forster's Tern", 'Royal Tern', 'Elegant Tern', 'Black Skimmer', 'Rock Dove', 'White-crowned Pigeon',
        'Red-billed Pigeon', 'Band-tailed Pigeon', 'Eurasian Collared Dove', 'African Collared Dove', 'Spotted Dove',
        'Zebra Dove', 'Inca Dove', 'Common Ground Dove', 'White-tipped Dove', 'White-winged Dove', 'Mourning Dove',
        'Mariana Fruit Dove', 'Pacific Imperial Pigeon', 'Smooth-billed Ani', 'Groove-billed Ani', 'Greater Roadrunner',
        'Yellow-billed Cuckoo', 'Mangrove Cuckoo', 'Black-billed Cuckoo', 'Pacific Long-tailed Cuckoo', 'Western Barn Owl',
        'Flammulated Owl', 'Western Screech Owl', 'Eastern Screech Owl', 'Whiskered Screech Owl', 'Great Horned Owl',
        'Northern Hawk-Owl', 'Mountain Pygmy Owl', 'Ferruginous Pygmy Owl', 'Elf Owl', 'Burrowing Owl', 'Spotted Owl',
        'Northern Barred Owl', 'Great Grey Owl', 'Long-eared Owl', 'Short-eared Owl', 'Boreal Owl', 'Northern Saw-whet Owl',
        'Lesser Nighthawk', 'Common Nighthawk', 'Antillean Nighthawk', 'Pauraque', 'Common Poorwill', "Chuck-will's-widow",
        'Buff-collared Nightjar', 'Eastern Whip-poor-will', 'Mexican Whip-poor-will', 'American Black Swift',
        'Chimney Swift', "Vaux's Swift", 'White-throated Swift', 'Magnificent Hummingbird', 'Plain-capped Starthroat',
        'Blue-throated Mountaingem', 'Lucifer Sheartail', 'Ruby-throated Hummingbird', 'Black-chinned Hummingbird',
        "Anna's Hummingbird", "Costa's Hummingbird", 'Broad-tailed Hummingbird', 'Rufous Hummingbird',
        "Allen's Hummingbird", 'Calliope Hummingbird', 'Broad-billed Hummingbird', 'Berylline Hummingbird',
        'Buff-bellied Hummingbird', 'Violet-crowned Hummingbird', 'Elegant Trogon', 'Collared Kingfisher',
        'Ringed Kingfisher', 'Belted Kingfisher', 'Green Kingfisher', "Lewis's Woodpecker", 'Red-headed Woodpecker',
        'Acorn Woodpecker', 'Gila Woodpecker', 'Golden-fronted Woodpecker', 'Red-bellied Woodpecker',
        "Williamson's Sapsucker", 'Yellow-bellied Sapsucker', 'Red-naped Sapsucker', 'Red-breasted Sapsucker',
        'Ladder-backed Woodpecker', "Nuttall's Woodpecker", 'Downy Woodpecker', 'Hairy Woodpecker', 'Arizona Woodpecker',
        'Red-cockaded Woodpecker', 'White-headed Woodpecker', 'American Three-toed Woodpecker', 'Black-backed Woodpecker',
        'Northern Flicker', 'Gilded Flicker', 'Pileated Woodpecker', 'Northern Crested Caracara', 'American Kestrel',
        'Merlin', 'Gyrfalcon', 'Peregrine Falcon', 'Prairie Falcon', 'Cockatiel', 'Rose-ringed Parakeet', 'Budgerigar',
        'Blue-crowned Lorikeet', 'Rosy-faced Lovebird', 'Monk Parakeet', 'White-winged Parakeet',
        'Yellow-chevroned Parakeet', 'Red-crowned Amazon', 'Lilac-crowned Amazon', 'Yellow-headed Amazon',
        'Orange-winged Amazon', 'Nanday Parakeet', 'Blue-and-yellow Macaw', 'Blue-crowned Parakeet', 'Green Parakeet',
        'Mitred Parakeet', 'Red-masked Parakeet', 'Northern Beardless Tyrannulet', 'Northern Tufted Flycatcher',
        'Olive-sided Flycatcher', 'Greater Pewee', 'Western Wood Pewee', 'Eastern Wood Pewee', 'Yellow-bellied Flycatcher',
        'Acadian Flycatcher', 'Alder Flycatcher', 'Willow Flycatcher', 'Least Flycatcher', "Hammond's Flycatcher",
        'American Grey Flycatcher', 'American Dusky Flycatcher', 'Pine Flycatcher', 'Pacific-slope Flycatcher',
        'Cordilleran Flycatcher', 'Buff-breasted Flycatcher', 'Black Phoebe', 'Eastern Phoebe', "Say's Phoebe",
        'Vermilion Flycatcher', 'Dusky-capped Flycatcher', 'Ash-throated Flycatcher', "Nutting's Flycatcher",
        'Great Crested Flycatcher', 'Brown-crested Flycatcher', "La Sagra's Flycatcher", 'Great Kiskadee',
        'Sulphur-bellied Flycatcher', 'Tropical Kingbird', "Couch's Kingbird", "Cassin's Kingbird", 'Thick-billed Kingbird',
        'Western Kingbird', 'Eastern Kingbird', 'Grey Kingbird', 'Scissor-tailed Flycatcher', 'Rose-throated Becard',
        'Micronesian Myzomela', 'Cardinal Myzomela', 'Wattled Honeyeater', 'Loggerhead Shrike', 'Great Grey Shrike',
        'Black-capped Vireo', 'White-eyed Vireo', "Bell's Vireo", 'Grey Vireo', "Hutton's Vireo", 'Yellow-throated Vireo',
        "Cassin's Vireo", 'Blue-headed Vireo', 'Plumbeous Vireo', 'Philadelphia Vireo', 'Warbling Vireo', 'Red-eyed Vireo',
        'Yellow-green Vireo', 'Black-whiskered Vireo', 'Hawaii Elepaio', 'Kauai Elepaio', 'Oahu Elepaio', 'Fiji Shrikebill',
        'Grey Jay', 'Black-throated Magpie-Jay', 'Brown Jay', 'Green Jay', 'Pinyon Jay', "Steller's Jay", 'Blue Jay',
        'Florida Scrub Jay', 'Island Scrub Jay', 'California Scrub Jay', "Woodhouse's Scrub Jay", 'Mexican Jay',
        'Black-billed Magpie', 'Yellow-billed Magpie', "Clark's Nutcracker", 'Mariana Crow', 'American Crow',
        'Northwestern Crow', 'Tamaulipas Crow', 'Fish Crow', 'Chihuahuan Raven', 'Northern Raven', 'Horned Lark',
        'Eurasian Skylark', 'Northern Rough-winged Swallow', 'Purple Martin', 'Tree Swallow', 'Violet-green Swallow',
        'Sand Martin', 'Barn Swallow', 'American Cliff Swallow', 'Cave Swallow', 'Carolina Chickadee',
        'Black-capped Chickadee', 'Mountain Chickadee', 'Mexican Chickadee', 'Chestnut-backed Chickadee',
        'Boreal Chickadee', 'Bridled Titmouse', 'Oak Titmouse', 'Juniper Titmouse', 'Tufted Titmouse',
        'Black-crested Titmouse', 'Verdin', 'American Bushtit', 'Red-breasted Nuthatch', 'White-breasted Nuthatch',
        'Pygmy Nuthatch', 'Brown-headed Nuthatch', 'Brown Creeper', 'Rock Wren', 'Canyon Wren', 'House Wren',
        'Pacific Wren', 'Winter Wren', 'Sedge Wren', 'Marsh Wren', 'Carolina Wren', "Bewick's Wren", 'Cactus Wren',
        'Sinaloa Wren', 'Blue-grey Gnatcatcher', 'California Gnatcatcher', 'Black-tailed Gnatcatcher',
        'Black-capped Gnatcatcher', 'American Dipper', 'Red-vented Bulbul', 'Red-whiskered Bulbul',
        'Golden-crowned Kinglet', 'Ruby-crowned Kinglet', 'Japanese Bush Warbler', 'Dusky Warbler', 'Arctic Warbler',
        'Saipan Reed Warbler', 'Millerbird', 'Wrentit', 'Japanese White-eye', 'Rota White-eye', 'Chinese Hwamei',
        'Red-billed Leiothrix', 'White-rumped Shama', 'Bluethroat', 'Northern Wheatear', 'Eastern Bluebird',
        'Western Bluebird', 'Mountain Bluebird', "Townsend's Solitaire", 'Brown-backed Solitaire', 'Omao', 'Puaiohi',
        'Orange-billed Nightingale-Thrush', 'Veery', 'Grey-cheeked Thrush', "Bicknell's Thrush", "Swainson's Thrush",
        'Hermit Thrush', 'Wood Thrush', 'Clay-colored Thrush', 'Rufous-backed Thrush', 'American Robin', 'Varied Thrush',
        'Grey Catbird', 'Curve-billed Thrasher', 'Brown Thrasher', 'Long-billed Thrasher', "Bendire's Thrasher",
        'California Thrasher', "Le Conte's Thrasher", 'Crissal Thrasher', 'Sage Thrasher', 'Northern Mockingbird',
        'Polynesian Starling', 'Samoan Starling', 'Common Hill Myna', 'Common Starling', 'Common Myna',
        'Eastern Yellow Wagtail', 'White Wagtail', 'Olive-backed Pipit', 'Red-throated Pipit', 'Buff-bellied Pipit',
        "Sprague's Pipit", 'Bohemian Waxwing', 'Cedar Waxwing', 'Phainopepla', 'Olive Warbler', 'Lapland Longspur',
        'Chestnut-collared Longspur', "Smith's Longspur", "McCown's Longspur", 'Snow Bunting', "McKay's Bunting",
        'Ovenbird', 'Worm-eating Warbler', 'Louisiana Waterthrush', 'Northern Waterthrush', 'Golden-winged Warbler',
        'Blue-winged Warbler', 'Black-and-white Warbler', 'Prothonotary Warbler', "Swainson's Warbler",
        'Crescent-chested Warbler', 'Tennessee Warbler', 'Orange-crowned Warbler', 'Colima Warbler', "Lucy's Warbler",
        'Nashville Warbler', "Virginia's Warbler", 'Connecticut Warbler', 'Grey-crowned Yellowthroat',
        "MacGillivray's Warbler", 'Mourning Warbler', 'Kentucky Warbler', 'Common Yellowthroat', 'Hooded Warbler',
        'American Redstart', "Kirtland's Warbler", 'Cape May Warbler', 'Cerulean Warbler', 'Northern Parula',
        'Tropical Parula', 'Magnolia Warbler', 'Bay-breasted Warbler', 'Blackburnian Warbler', 'Mangrove Warbler',
        'Chestnut-sided Warbler', 'Blackpoll Warbler', 'Black-throated Blue Warbler', 'Palm Warbler', 'Pine Warbler',
        'Myrtle Warbler', 'Yellow-throated Warbler', 'Prairie Warbler', "Grace's Warbler", 'Black-throated Grey Warbler',
        "Townsend's Warbler", 'Hermit Warbler', 'Golden-cheeked Warbler', 'Black-throated Green Warbler',
        'Rufous-capped Warbler', 'Golden-crowned Warbler', 'Canada Warbler', "Wilson's Warbler", 'Red-faced Warbler',
        'Painted Whitestart', 'Slate-throated Whitestart', 'Red-crested Cardinal', 'Saffron Finch',
        'White-collared Seedeater', 'Bananaquit', 'Rufous-winged Sparrow', "Botteri's Sparrow", "Cassin's Sparrow",
        "Bachman's Sparrow", 'Grasshopper Sparrow', "Baird's Sparrow", "Henslow's Sparrow", "Le Conte's Sparrow",
        "Nelson's Sparrow", 'Saltmarsh Sparrow', 'Seaside Sparrow', 'Olive Sparrow', 'American Tree Sparrow',
        'Chipping Sparrow', 'Clay-colored Sparrow', 'Black-chinned Sparrow', 'Field Sparrow', "Brewer's Sparrow",
        'Black-throated Sparrow', 'Five-striped Sparrow', 'Lark Sparrow', 'Lark Bunting', 'Fox Sparrow', 'Dark-eyed Junco',
        'Yellow-eyed Junco', 'White-crowned Sparrow', 'Golden-crowned Sparrow', "Harris's Sparrow",
        'White-throated Sparrow', 'Sagebrush Sparrow', "Bell's Sparrow", 'Vesper Sparrow', 'Savannah Sparrow',
        'Song Sparrow', "Lincoln's Sparrow", 'Swamp Sparrow', 'Canyon Towhee', "Abert's Towhee", 'California Towhee',
        'Rufous-crowned Sparrow', 'Green-tailed Towhee', 'Spotted Towhee', 'Eastern Towhee', 'Western Spindalis',
        'Yellow-breasted Chat', 'Hepatic Tanager', 'Summer Tanager', 'Scarlet Tanager', 'Western Tanager',
        'Flame-colored Tanager', 'Crimson-collared Grosbeak', 'Northern Cardinal', 'Pyrrhuloxia', 'Rose-breasted Grosbeak',
        'Black-headed Grosbeak', 'Blue Grosbeak', 'Lazuli Bunting', 'Indigo Bunting', 'Varied Bunting', 'Painted Bunting',
        'Dickcissel', 'Yellow-headed Blackbird', 'Bobolink', 'Western Meadowlark', 'Eastern Meadowlark',
        'Puerto Rican Oriole', 'Black-vented Oriole', 'Orchard Oriole', 'Hooded Oriole', 'Streak-backed Oriole',
        "Bullock's Oriole", 'Spot-breasted Oriole', 'Altamira Oriole', "Audubon's Oriole", 'Baltimore Oriole',
        "Scott's Oriole", 'Red-winged Blackbird', 'Tricolored Blackbird', 'Shiny Cowbird', 'Bronzed Cowbird',
        'Brown-headed Cowbird', 'Rusty Blackbird', "Brewer's Blackbird", 'Common Grackle', 'Boat-tailed Grackle',
        'Great-tailed Grackle', 'Evening Grosbeak', 'Akikiki', 'Maui Alauahio', 'Palila', 'Nihoa Finch', 'Akohekohe',
        'Apapane', 'Iiwi', 'Maui Parrotbill', 'Akiapolaau', 'Anianiau', 'Hawaii Amakihi', 'Oahu Amakihi', 'Kauai Amakihi',
        'Hawaii Creeper', 'Akekee', 'Akepa', 'Pine Grosbeak', 'Grey-crowned Rosy Finch', 'Black Rosy Finch',
        'Brown-capped Rosy Finch', 'House Finch', 'Purple Finch', "Cassin's Finch", 'Yellow-fronted Canary',
        'Common Redpoll', 'Arctic Redpoll', 'Red Crossbill', 'Two-barred Crossbill', 'Pine Siskin', 'Lesser Goldfinch',
        "Lawrence's Goldfinch", 'American Goldfinch', 'House Sparrow', 'Eurasian Tree Sparrow', 'Common Waxbill',
        'Red Avadavat', 'Bronze Mannikin', 'African Silverbill', 'Scaly-breasted Munia', 'Chestnut Munia', 'Java Sparrow',
        'Pin-tailed Whydah',
    ]),

)
