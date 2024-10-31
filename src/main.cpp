#include <string>

#include <SFML/Graphics.hpp>

#include "elements/button.hpp"
#include "elements/tabs/system.hpp"
#include "config.hpp"


TabSystem* build_ui() {
    TabSystem* tabs = new TabSystem(10.f, sf::Color::Green);
    Tab* tab = tabs->addTab("Overview");
    tab = tabs->addTab("AI modules influences");
    // tab->AddElement(new Button(
    //     100, 100, "Help!",
    //     PATH_FONT_DEFAULT, sf::Color::Green
    // ));
    return tabs;
}

int main() {
    sf::RenderWindow window(
        sf::VideoMode(800, 600),
        MAICLient_NAME,
        sf::Style::Titlebar |
        sf::Style::Close
    );
    ROOT_WINDOW = &window;
    TabSystem* sys = build_ui();
    window.setFramerateLimit(60);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
                break;
            }
            sys->handleEvent(event);
        }

        window.clear();
        sys->draw(window);
        window.display();
    }

    delete sys;
    return 0;
}
