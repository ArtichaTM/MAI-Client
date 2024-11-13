#include "./header.hpp"
#include "elements/list/vertical.hpp"
#include "elements/interactive/button.hpp"

void add_overview(TabSystem* sys) {
    Tab* tab = sys->addTab("Overview");
    // sf::FloatRect temp_bounds(tab->getGlobalBounds());
    // tab->AddElement(
    //     new VerticalList(0, temp_bounds.top + temp_bounds.height)
    //         ->addElement(
    //             new Button()
    //         )
    // );
}
