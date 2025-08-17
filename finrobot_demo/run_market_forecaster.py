import autogen
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant


def forecast_ticker(ticker: str, out_path: str) -> None:
	llm_config = {
		"config_list": autogen.config_list_from_json(
			"./OAI_CONFIG_LIST",
			filter_dict={"model": ["gpt-5-mini-2025-08-07"]},
		),
		"timeout": 180,
		"temperature": 0,
	}
	register_keys_from_json("./config_api_keys")

	assistant = SingleAssistant(
		"Market_Analyst",
		llm_config,
		human_input_mode="NEVER",
	)

	message = (
		f"Use all the tools provided to retrieve information available for {ticker} upon {get_current_date()}. "
		"Analyze the positive developments and potential concerns of {ticker} with 2-4 most important factors respectively and keep them concise. "
		"Most factors should be inferred from company related news. "
		f"Then make a rough prediction (e.g. up/down by 2-3%) of the {ticker} stock price movement for next week. "
		"Provide a summary analysis to support your prediction."
	)

	# Capture stdout by using autogen's conversation summary via cache, and also tee console output to file
	import sys
	from io import StringIO

	buffer = StringIO()
	old_stdout = sys.stdout
	sys.stdout = buffer
	try:
		assistant.chat(message)
	finally:
		sys.stdout = old_stdout

	with open(out_path, "w", encoding="utf-8") as f:
		f.write(buffer.getvalue())
	print(f"Saved forecast report for {ticker} to {out_path}")


if __name__ == "__main__":
	forecast_ticker("MSFT", "./report/forecast_MSFT.txt")
	forecast_ticker("NVDA", "./report/forecast_NVDA.txt")