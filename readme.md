# GetPeak

------------------------------------
GetPeak is a tool for spectrum data analysis. It is free to use for scientific purposes.
**This Software is under development. Any Feedback will be valuable. For questions or requests, please contact Y.S.Murakami at sterling.astro@berkeley.edu. Thank you!**

## Dependency
Tested with Python>=3.6 and requires ```matplotlib```,```scipy```,and ```numpy```.

## Installation
At this version, installation is not required. The quickest usage is

``` from getpeak import do_analysis ```

and then

``` do_analysis('filename.dat') ```.

the internally called function ``` get_data() ``` needs to be modified for each data format. To check current data format, see sampele data in ```/data/``` folder.

## Development Plan
- Introduce GUI with Bokeh
- Introduce multi-band detection (iteration through multiple sampling width)

