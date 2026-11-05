function h = plot_mustang(x, scale, geometry)
%PLOT_MUSTANG Plot an eigenvector as a deformed aircraft.
%   scale is the maximum displayed displacement in metres.
%   scale = 0 displays the undeformed aircraft.

    U = real(x(geometry.displacement_dofs)).';
    magnitude = sqrt(sum(U.^2, 2));
    umax = max(magnitude);

    if umax == 0
        umax=eps()
    end

    U = U / umax;

    figure('Color', 'w');
    h = patch('Faces', geometry.faces, ...
              'Vertices', geometry.points + scale*U, ...
              'FaceVertexCData', magnitude/umax, ...
              'FaceColor', 'interp', ...
              'EdgeColor', 'none');

    axis equal
    axis tight
    view(3)
    xlabel('x (m)');
    ylabel('y (m)');
    zlabel('z (m)');
    colormap(parula);
    caxis([0 1]);
    cb = colorbar;
    cb.Label.String = 'Normalized displacement magnitude';
    camlight headlight
    lighting gouraud
    rotate3d on
end