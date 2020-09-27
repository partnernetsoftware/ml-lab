
# References

* https://blog.csdn.net/pansaky/article/details/89335977

```
IEX
Warning

Usage of all IEX readers now requires an API key. See below for additional information.

The Investors Exchange (IEX) provides a wide range of data through an API. Historical stock prices are available for up to 15 years. The usage of these readers requires the publishable API key from IEX Cloud Console, which can be stored in the IEX_API_KEY environment variable.

In [1]: import pandas_datareader.data as web

In [2]: from datetime import datetime

In [3]: start = datetime(2016, 9, 1)

In [4]: end = datetime(2018, 9, 1)

In [5]: f = web.DataReader('F', 'iex', start, end)

In [6]: f.loc['2018-08-31']
Out[6]:
open             9.64
high             9.68
low              9.40
close            9.48
volume    76424884.00
Name: 2018-08-31, dtype: float64
Note

You must provide an API Key when using IEX. You can do this using os.environ["IEX_API_KEY"] = "pk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" or by exporting the key before starting the IPython session.

There are additional interfaces to this API that are directly exposed: tops (‘iex-tops’) and last (‘iex-lasts’). A third interface to the deep API is exposed through Deep class or the get_iex_book function.

Todo

Execute block when markets are open
```
