In the realm of Digital Marketing, personalizing sales and contacts basing on individual customers with Data Driven strategies is becoming increasingly important. It is essential to identify the target to address a specific product/offer through the most suitable channel. Advanced Analytics algorithms can support rules-based tools in making the personalization of marketing/caring campaigns more accurate.

The present work analyzes data of a company that deals with the sale of electricity, gas, and high energy efficiency solution (boilers, air conditioners, photovoltaics).

The goal is to develop, on an annual basis, a monthly contact strategy that maximizes the success of different marketing campaigns by avoiding contacting the customer excessively and distributing the contacts evenly over.

Specifically, there are two campaigns the company proposes: “cross-selling”, to offer a commodity to a customer that has a gas/power contract, and “solution”, to offer a highly energy efficient solution. These, in turn, are promoted through three communication channels: Direct Email Marketing (DEM), SMS and Teleselling (TLS).

Therefore, the monthly contact strategy will take the form of 6 csv files, each for every possible combination of marketing campaign and communication channel. In particular, the files must have one row per customer, and contain the following columns:

• ID: The customer’s ID;

• Month 1, Month 2,...Month 12: they will take the value 0 if the customer will not be contacted
  for the campaign with that channel and 1 otherwise.

To fully achieve the desired result, there were two models to be implemented: a propensity model, to estimate the likelihood of customers’ positive responses to a specific campaign, and an eligibility model, to analyze the customers’ contactability.

Together these two models would allow to assign to each customer a campaign and a contact channel through which the company can send successful marketing communications.
