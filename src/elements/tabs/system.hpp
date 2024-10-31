#pragma once
#include <vector>
#include <string>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "tab.hpp"
#include "../base.hpp"


class TabSystem : public SFBase {
    std::vector<Tab*> tabs;
    size_t active_tab_index = 0;
    sf::Font font;
    const float height;
    const sf::Color color;
    static constexpr float offset = 50.f;
public:
    TabSystem(float height, sf::Color color);
    ~TabSystem();
    void handleEvent(const sf::Event&) override;
    void draw(sf::RenderWindow& window) const override;
    Tab* addTab(const std::string& title);

    Tab* operator[](const std::string& name);
    Tab* operator[](ushort index);
};
