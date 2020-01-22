
def create_zipf_table(frequencies):

    """
    Takes the list created by _top_word_frequencies
    and inserts it into a list of dictionaries,
    along with the Zipfian data.
    """

    zipf_table = []

    top_frequency = frequencies[0][1]

    for index, item in enumerate(frequencies, start=1):

        relative_frequency = "1/{}".format(index)
        zipf_frequency = top_frequency * (1 / index)
        difference_actual = item[1] - zipf_frequency
        difference_percent = (item[1] / zipf_frequency) * 100

        zipf_table.append({"word": item[0],
                           "actual_frequency": item[1],
                           "relative_frequency": relative_frequency,
                           "zipf_frequency": zipf_frequency,
                           "difference_actual": difference_actual,
                           "difference_percent": difference_percent})

    return zipf_table


def print_zipf_table(zipf_table, n):

    """
    Prints the list created by generate_zipf_table
    in table format with column headings.
    """

    width = 80

    print("-" * width)
    print("|Rank|    Word    |Actual Freq | Zipf Frac  | Zipf Freq  |Actual Diff |Pct Diff|")
    print("-" * width)

    format_string = "|{:4}|{:12}|{:12.0f}|{:>12}|{:12.2f}|{:12.2f}|{:7.2f}%|"

    for index, item in enumerate(zipf_table[0:n], start=1):

        print(format_string.format(index,
                                   item["word"],
                                   item["actual_frequency"],
                                   item["relative_frequency"],
                                   item["zipf_frequency"],
                                   item["difference_actual"],
                                   item["difference_percent"]))

    print("-" * width)
	
	
def plot_zipfs_graph(zipf_table, freq_dist, n):
	"""
	Plots 2 graphs:
	graph 1: log(f) as a function of log(r)
	graph 2:  word count as a function of word
	"""
	
	from math import log2
	import matplotlib.pyplot as plt
	
	log_r = []
	log_f = []
	for index, item in enumerate(zipf_table, start=1):
		freq = item['actual_frequency']
		log_r.append(log2(index))
		log_f.append(log2(freq))

	fig = plt.figure(figsize = (6,8))
	plt.subplots_adjust(hspace = 0.5)

	ax1 = plt.subplot(2,1,1)
	plt.plot(log_r, log_f, 'ro')
	plt.suptitle("Zipf's Law", fontsize=20)
	plt.xlabel('log(r)', fontsize=16)
	plt.ylabel('log(f)', fontsize=16)

	ax2 = plt.subplot(2,1,2)
	freq_dist.plot(n, cumulative=False)
	plt.show()