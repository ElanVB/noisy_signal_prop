docker run -it -v $(pwd):/wd -w /wd signal_prop:latex bash -c "cd src; python simulate.py;"
# docker run -it -v $(pwd):/wd -w /wd signal_prop:plot bash -c "cd src; python plotting.py;"
