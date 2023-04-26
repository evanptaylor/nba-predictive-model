# nba-predictive-model

model.py is a predictive model for the outright winner of NBA games that uses machine learning to predict the outcomes of NBA games based on historical game data and upcoming Vegas odds from DraftKings. The model combines a Random Forest Classifier, Logistic Regression, and Gradient Boost Classifier using a Voting Classifier to find the best fit and generate predictions. Right now I train the model using historical data from after the trade deadline since I haven't added player stats yet. The predictions are loaded into an Excel file (predictions.xlsx) for readability. I also included certified locks--winners that have have a high probability (>0.75) yet better than expected odds.

To run the program you'll need your own API key from the-odds-api.com

##### Example output for some upcoming first-round playoff games:
![Screen Shot 2023-04-26 at 4 55 37 PM](https://user-images.githubusercontent.com/36122439/234700835-36a8236a-4863-4db9-8549-372fa5556323.png)
