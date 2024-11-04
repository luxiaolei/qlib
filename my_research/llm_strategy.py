from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
import pandas as pd


class LLMStrategy(BaseSignalStrategy):
    """Strategy that converts LLM predictions into trading decisions."""
    
    def __init__(
        self,
        signal,  # This will be the predictions DataFrame
        position_size: float = 1.0,
        **kwargs
    ):
        """Initialize strategy.
        
        Parameters
        ----------
        signal : DataFrame
            The predictions DataFrame
        position_size : float
            Size of position to take for each trade
        """
        super().__init__(signal=signal, **kwargs)  # BaseSignalStrategy will handle creating Signal object
        self.position_size = position_size
        self.trade_positions = {}

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        """Generate trading decisions based on predictions.
        
        Parameters
        ----------
        execute_result : List[Tuple], optional
            The execution result of previous decision.

        Returns
        -------
        TradeDecisionWO
            The trade decision with generated orders.
        """
        # Get current trading step info
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        # Get signal predictions
        pred_scores = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_scores is None:
            return TradeDecisionWO(order_list=[], strategy=self)

        # Update positions based on execution results
        if execute_result is not None:
            for order, _, deal_amount, _ in execute_result:
                self.trade_positions[order.stock_id] = deal_amount if order.direction == OrderDir.BUY else -deal_amount

        order_list = []

        # Process each stock prediction using iterrows()
        for stock_id, row in pred_scores.iterrows():
            pred_score = row['score']  # Get the score value from the row
            
            # Skip if prediction is missing or stock is not tradable
            if pd.isna(pred_score) or not self.trade_exchange.is_stock_tradable(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time
            ):
                continue

            # Determine trade direction based on prediction
            direction = OrderDir.BUY if pred_score == 1 else OrderDir.SELL

            # Calculate trade amount
            amount_unit = self.trade_exchange.get_amount_of_trade_unit(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time
            )

            if amount_unit is None:
                # If no trade unit specified, use position size directly
                trade_amount = self.position_size
            else:
                # Round to nearest trade unit
                trade_amount = round(self.position_size / amount_unit) * amount_unit

            # Create order
            if trade_amount > 0:
                order = Order(
                    stock_id=stock_id,
                    amount=trade_amount,
                    direction=direction,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
                order_list.append(order)

        return TradeDecisionWO(order_list=order_list, strategy=self) 