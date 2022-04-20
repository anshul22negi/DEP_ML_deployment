import json
with open('classes.json') as json_file:
    classes = json.load(json_file)

classes = {int(k):v for k,v in classes.items()}
print(classes)