# docker run -it -v $(pwd):/wd -w /wd signal_prop:latex bash -c "cd src; python plotting.py;"
docker run -it -v $(pwd):/wd -w /wd signal_prop:latex bash -c "cd src; python plot_theory.py;"
