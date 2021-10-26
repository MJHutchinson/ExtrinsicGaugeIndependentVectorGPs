import requests
import os
import click
from zipfile import ZipFile

@click.command()
@click.option("--target-dir", default = "weatherbench_wind_data", type=str, help="directory to save data")
@click.option("--resolution", default = 5, type=int, help="resolution of wind data (choose from 1, 2 or 5)")
def get_winddata(target_dir, resolution):
    """
    This function downloads the weatherbench regridded ERA5 global wind data from 1979-2018
    Args:
        target_dir: target directory to save wind data
        resolution: resolution of wind data. To choose between 1, 2 or 5.
    """
    if resolution == 1:
        res = 1.40625
    elif resolution == 2:
        res = 2.8125
    elif resolution == 5:
        res = 5.625
    else:
        raise ValueError("Resolution must be either 1, 2 or 5")

    url_u = f"https://dataserv.ub.tum.de/s/m1524895/download?path=%2F{res}deg%2F10m_u_component_of_wind&files=10m_u_component_of_wind_{res}deg.zip"
    url_v = f"https://dataserv.ub.tum.de/s/m1524895/download?path=%2F{res}deg%2F10m_v_component_of_wind&files=10m_v_component_of_wind_{res}deg.zip"

    # Download u and v components of global wind
    print("fetching url for u component...")
    r_u = requests.get(url_u, verify=False)
    print("fetching url for v component...")
    r_v = requests.get(url_v, verify=False)

    zipfile_u = f"10m_u_component_of_wind_{res}deg.zip"
    zipfile_v = f"10m_v_component_of_wind_{res}deg.zip"

    print("saving contents of u component...")
    open(zipfile_u, 'wb').write(r_u.content)
    print("saving contents of v component...")
    open(zipfile_v, 'wb').write(r_v.content)
    
    os.mkdir(target_dir) if not os.path.exists(target_dir) else None

    # Unzip files to target directory
    print("unzipping file for u component...")
    with ZipFile(zipfile_u, 'r') as zipObj:
        zipObj.extractall(target_dir)
    print("unzipping file for v component...")
    with ZipFile(zipfile_v, 'r') as zipObj:
        zipObj.extractall(target_dir)
    
    # Delete zip files
    print("delete zip files...")
    os.remove(zipfile_u)
    os.remove(zipfile_v)

    print("complete! data saved on: " + target_dir)


if __name__ == '__main__':
    get_winddata()
