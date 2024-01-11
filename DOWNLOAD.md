Dataset **TAS500** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/O/7/Aj/aIRGnDsz9PHdA98T4Lhr2aaK2tqyxcwunWBvUCTdpf6KEAk1rrZVWEHgFu3wk4S2y32N2eyTQhuChWNxA1N1lqUUmA64kej4nqVDNDUKAprY74qEQjZgIpWxJ8mB.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='TAS500', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://drive.google.com/open?id=1dCxJNWGbQT-tbuDTD6nQvmWPwGS9MPwT&authuser=0).