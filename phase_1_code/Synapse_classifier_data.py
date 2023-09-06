filepath = "data/Synapse/formatted_crc_data.txt"

# display first 10 lines
with open(filepath) as fp:
    for i, line in enumerate(fp):
        print(line)
        if i == 1:
            break
