import numpy as np

class StockTradingEnv:
    def __innit__(self, df):
        self.df = df
        self.reset()

        self.action_space_n = 11
        self.observation_space_shape = (len(self.df.columns),)
    def reset(self):

        self.current_step = 0
        self.account_balance = 10000
        self.shares_held = 0
        self.net_worth = self.account_balance
        self.prev_net_worth = self.account_balance
        self.max_net_worth = self.account_balance
        return self._next_observation()
    
    def _next_observation(self):
        #numpy array of current market conditions
        obs = self.df.iloc[self.current_step].values
        return obs
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        done = False
        
        if action == 0: #sell 100% of stocks
            if self.shares_held > 0:
                sale_value = self.shares_held * current_price
                self.account_balance += sale_value
                self.shares_held = 0
        elif action == 1: #sell 80% of stocks
            if self.shares_held > 0:
                shares_to_sell = int(self.shares_held * 0.8)
                if shares_to_sell > 0:
                    sale_value = shares_to_sell * current_price
                    self.account_balance += sale_value
                    self.shares_held -= shares_to_sell
        elif action == 2: #sell 60% of stocks
            if self.shares_held > 0:
                shares_to_sell = int(self.shares_held * 0.6)
                if shares_to_sell > 0:
                    sale_value = shares_to_sell * current_price
                    self.account_balance += sale_value
                    self.shares_held -= shares_to_sell
        elif action == 3: #sell 40% of stocks
            if self.shares_held > 0:
                shares_to_sell = int(self.shares_held * 0.4)
                if shares_to_sell > 0:
                    sale_value = shares_to_sell * current_price
                    self.account_balance += sale_value
                    self.shares_held -= shares_to_sell
        elif action == 4: #sell 20% of stocks
            if self.shares_held > 0:
                shares_to_sell = int(self.shares_held * 0.2)
                if shares_to_sell > 0:
                    sale_value = shares_to_sell * current_price
                    self.account_balance += sale_value
                    self.shares_held -= shares_to_sell
        elif action == 5:#hold shares
            pass
        elif action == 6: #buy 20% of cash available amount of stocks
            if self.account_balance > 0:
                cash_available = self.account_balance * 0.2
                shares_to_buy = cash_available / current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.shares_held += shares_to_buy
                    self.account_balance -= cost
        elif action == 7:
            if self.account_balance > 0: #buy 40% of cash available amount of stocks
                cash_available = self.account_balance * 0.4
                shares_to_buy = cash_available / current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.shares_held += shares_to_buy
                    self.account_balance -= cost
        elif action == 8: #buy 60% of cash available amount of stocks
            if self.account_balance > 0:
                cash_available = self.account_balance * 0.6
                shares_to_buy = cash_available / current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.shares_held += shares_to_buy
                    self.account_balance -= cost
        elif action == 9: #buy 80% of cash available amount of stocks
            if self.account_balance > 0:
                cash_available = self.account_balance * 0.8
                shares_to_buy = cash_available / current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.shares_held += shares_to_buy
                    self.account_balance -= cost
        elif action == 10: #buy 100% of cash available amount of stocks
            if self.account_balance > 0:
                cash_available = self.account_balance
                shares_to_buy = cash_available / current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.shares_held += shares_to_buy
                    self.account_balance -= cost

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
        
        reward = self.net_worth - self.prev_net_worth

        obs = self._next_observation()

        return obs, reward, done, {}
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
