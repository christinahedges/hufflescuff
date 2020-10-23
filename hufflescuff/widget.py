import numpy as np

import bokeh
from bokeh.plotting import figure, ColumnDataSource, save
from bokeh.io import output_notebook, show, output_file
from bokeh.layouts import layout, Spacer, row
from bokeh.models import LinearColorMapper, MultiLine

def prepare_img_datasource(img, theta, rho):
    npix = img.size
    source = ColumnDataSource(data=dict(xx=theta.ravel(), yy=rho.ravel()))
    source.selected.indices = [np.argmax(img)]
    return source

def make_img_fig(img, theta, rho, source):
    fig = figure(plot_width=400, plot_height=400,
             x_range=(theta.min(), theta.max()),
             y_range=(rho.min(), rho.max()),
             title='Hough Transform', tools='tap,box_select,wheel_zoom,reset',
             toolbar_location="below",
             border_fill_color="whitesmoke")
    fig.image([img], x=theta.min(), y=rho.min(),
              dw=theta.max() - theta.min(), dh=rho.max() - rho.min(),
             color_mapper=LinearColorMapper(palette='Viridis256', low=img.min(), high=img.max()))
    fig.rect('xx', 'yy', np.median(np.diff(theta[0])), np.median(np.diff(rho[:, 0])), source=source, fill_color='gray',
                fill_alpha=0.1, line_color='white')
    return fig

def make_scat_fig(x, y):
    fig = figure(plot_width=400, plot_height=400,
         x_range=(x.min(), x.max()),
         y_range=(y.min(), y.max()),
         title='Points', tools='tap,box_select,wheel_zoom,reset',
         toolbar_location="below",
         border_fill_color="whitesmoke")
    fig.scatter(x, y, color='black', size=1)
    return fig

def show_widget(x, y, notebook_url='localhost:8888', ntheta=80, nrho=121):
    """ Shows the Hough Transform widget

    Paramters
    ---------
    x : np.ndarray
        Input x values for transform
    y : np.ndarray
        Input y values for transform
    notebook_url : str
        The jupyter notebook local host URL you are using
    ntheta : int
        Number of bins in the theta dimension
    nrho :
        Number of bins in the rho dimension
    """
    def hough_transform(x, y, ntheta=80, nrho=121):
        theta = np.deg2rad(np.linspace(0, 180, ntheta + 1)[:-1])

        diag_len = np.hypot(x, y).max()
        dr = np.linspace(-diag_len, diag_len, nrho)

        # Cache some resuable values
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        hough = np.zeros((nrho, ntheta), dtype=np.uint64)
        for x_idx, y_idx in zip(x, y):
            rho = (x_idx * cos_t + y_idx * sin_t)
            rho_loc = np.asarray([np.argmin(np.abs(dr - r1)) for r1 in rho])
            hough[rho_loc, np.arange(ntheta)] += 1
        Theta, Rho = np.meshgrid(theta, dr)
        return hough, Theta, Rho

    img, theta, rho = hough_transform(x, y, ntheta=ntheta, nrho=nrho)
    xp = np.linspace(x.min(), x.max(), 2)


    def calc_new_line(locs):
        xps = []
        yps = []
        for loc in locs:
            loc = np.unravel_index(loc, img.shape)
            rho1, theta1 = rho[:, 0][loc[0]], theta[0][loc[1]]
            yp = -np.cos(theta1)/np.sin(theta1) * xp + rho1/np.sin(theta1)
            xps.append(xp.tolist())
            yps.append(yp.tolist())
            xps.append(np.nan)
            yps.append(np.nan)
        return np.hstack(xps), np.hstack(yps)

    def create_interact_ui(doc):
        # The data source includes metadata for hover-over tooltips
        source = prepare_img_datasource(img, theta, rho)
        locs = [np.argmax(img)]

        xps, yps = calc_new_line(locs)
        line_source = ColumnDataSource(data=dict(xs=xps, ys=yps))

        # Create the lightcurve figure and its vertical marker
        fig = make_img_fig(img, theta, rho, source)

        fig_scat = make_scat_fig(x, y)
        fig_scat.line(x='xs', y='ys', line_color='red', source=line_source)

        p = layout([[fig_scat, fig]])
        doc.add_root(p)

        def update_upon_pixel_selection(attr, old, new):
            """Callback to take action when pixels are selected."""
            xps, yps = calc_new_line(new)
            line_source.data['xs'] = xps
            line_source.data['ys'] = yps

        source.selected.on_change('indices', update_upon_pixel_selection)
    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)
