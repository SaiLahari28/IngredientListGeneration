# Ingredients list generation from food images using multi-class classification and seq2seq modelling

Ingredients list generation takes place in 2 steps. The model architecture can be seen in the following image:

<img width="600" alt="image" src="https://github.com/SaiLahari28/IngredientListGeneration/assets/133794318/a5d20ab7-cd50-4a71-b333-9cd3566d42f1">

Dataset used for training the multi-class classification model(EfficientNet-B2) which predicts the food name can be found at: https://www.kaggle.com/datasets/synysterjeet/food-classification

Dataset used for training the text generation model(Transformer) which generates the ingredients list can be found at: https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags

Using the preprocess.py file data preprocessing of the dataset for the transformer is done. The final preprocessed data is also uploaded in the repository as dataPreprocessed.p.

Steps to train and run the model:

1. Run efficientnet.py to train the classifier model. After running this find .pth file of the classifier model. Rename the best model to efnet.pth.
2. Download dataPreprocessed.p and classList files. If wanted you can also preprocess the food ingredients data using the preprocess.py file but the final preprocessed data has been uploaded.
3. Run the project.ipynb notebook to train the transformer and generate ingredients list using both the classifier model and transformer model as done on the sample images in the notebook.
