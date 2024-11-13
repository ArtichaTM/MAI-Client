#include <iostream>

#include "./header.hpp"
#include "config.hpp"
#include "elements/list/vertical.hpp"
#include "elements/interactive/button.hpp"

void add_overview(TabSystem* sys) {
    Tab* tab = sys->addTab("Overview");
    sf::FloatRect temp_bounds(tab->getGlobalBounds());
    tab->AddElement(
        (new VerticalList(0, temp_bounds.top + temp_bounds.height))
            ->addElement(
                (new Button(2., 2., "Run", PATH_FONT_DEFAULT, sf::Color::Magenta))
                    ->setOnClick([](Button* button) {
                        std::cout << "Running!" << std::endl;
                    })
                    ->toggleActivatable()
            )
    );
}
