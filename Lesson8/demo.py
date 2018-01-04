# import libraries
import pandas as pd

# read tha annotations file
new_data = pd.read_csv('data/annotations_final.csv', sep="\t")

print(new_data.head())
print(new_data.info())

# generalize similar featueres for simplicity
# Some of the tags in the dataset are really close to each other. Lets merge them together
synonyms = [['beat', 'beats'],
            ['chant', 'chanting'],
            ['choir', 'choral'],
            ['classical', 'clasical', 'classic'],
            ['drum', 'drums'],
            ['electro', 'electronic', 'electronica', 'electric'],
            ['fast', 'fast beat', 'quick'],
            ['female', 'female singer', 'female singing', 'female vocals', 'female vocal', 'female voice', 'woman',
             'woman singing', 'women'],
            ['flute', 'flutes'],
            ['guitar', 'guitars'],
            ['hard', 'hard rock'],
            ['harpsichord', 'harpsicord'],
            ['heavy', 'heavy metal', 'metal'],
            ['horn', 'horns'],
            ['india', 'indian'],
            ['jazz', 'jazzy'],
            ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
            ['no beat', 'no drums'],
            ['no singer', 'no singing', 'no vocal', 'no vocals', 'no voice', 'no voices', 'instrumental'],
            ['opera', 'operatic'],
            ['orchestra', 'orchestral'],
            ['quiet', 'silence'],
            ['singer', 'singing'],
            ['space', 'spacey'],
            ['string', 'strings'],
            ['synth', 'synthesizer'],
            ['violin', 'violins'],
            ['vocal', 'vocals', 'voice', 'voices'],
            ['strange', 'weird']]

# Merge the synonyms and drop all other columns than the first one.
"""
Example:
Merge 'beat', 'beats' and save it to 'beat'.
Merge 'classical', 'clasical', 'classic' and save it to 'classical'.
"""
for synonym_list in synonyms:
    new_data[synonym_list[0]] = new_data[synonym_list].max(axis=1)
    new_data.drop(synonym_list[1:], axis=1, inplace=True)

# Drop the mp3_path tag from the dataframe
new_data.drop('mp3_path', axis=1, inplace=True)

print(new_data.head())
print(new_data.info())

# split data
training_set = new_data[:19773]
validation_set = new_data[19773:21294]
testing_set = new_data[21294:]