Yes
# ============================================
# POVERTY MAPS PAGE (SIMPLIFIED & WORKING)
# ============================================

elif page == "Poverty Maps":
    st.markdown('<p class="sub-header">📊 Poverty Distribution in Nigeria</p>', unsafe_allow_html=True)
    st.markdown("Visualizations showing poverty distribution and related factors across Nigeria")
    
    map_type = st.selectbox(
        "Select Indicator",
        ["Poverty Rate", "Flood Risk", "Conflict Level", "Internet Penetration", "Farming Dependency"]
    )
    
    map_data = context_data.copy()
    
    # Sort data by the selected indicator
    if map_type in ["Poverty Rate", "Internet Penetration", "Farming Dependency"]:
        sorted_data = map_data.sort_values(map_type, ascending=False)
    else:
        # For categorical data, sort by State
        sorted_data = map_data.sort_values('State')
    
    # Create two columns: chart and rankings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{map_type} by State")
        
        if map_type in ["Poverty Rate", "Internet Penetration", "Farming Dependency"]:
            # For numerical data: show bar chart
            top_15 = sorted_data.head(15)
            
            # Color based on value
            if map_type == "Poverty Rate":
                color_scale = 'Reds'
                title_suffix = "(Higher = More Poverty)"
            elif map_type == "Internet Penetration":
                color_scale = 'Blues'
                title_suffix = "(Higher = Better Access)"
            else:
                color_scale = 'Greens'
                title_suffix = "(Higher = More Dependent)"
            
            fig = px.bar(
                top_15,
                x=map_type,
                y='State',
                orientation='h',
                title=f"{map_type} {title_suffix}",
                color=map_type,
                color_continuous_scale=color_scale,
                height=500,
                text=map_type
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(
                xaxis_title=map_type,
                yaxis_title="State",
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # For categorical data: show pie chart and distribution
            fig = px.pie(
                sorted_data,
                names=map_type,
                title=f"{map_type} Distribution Across States",
                color=map_type,
                color_discrete_sequence=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'],
                height=450
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 State Rankings")
        st.markdown("---")
        
        if map_type in ["Poverty Rate", "Internet Penetration", "Farming Dependency"]:
            # Show top and bottom states
            top_10 = sorted_data.head(10)[['State', map_type]]
            st.markdown("**🔴 Highest Values:**")
            st.dataframe(top_10.set_index('State'), use_container_width=True)
            
            bottom_10 = sorted_data.tail(10)[['State', map_type]]
            st.markdown("**🟢 Lowest Values:**")
            st.dataframe(bottom_10.set_index('State'), use_container_width=True)
            
            # Show summary statistics
            st.markdown("---")
            st.markdown("**📊 Summary Statistics:**")
            stats = {
                "Mean": map_data[map_type].mean(),
                "Median": map_data[map_type].median(),
                "Min": map_data[map_type].min(),
                "Max": map_data[map_type].max(),
                "Std Dev": map_data[map_type].std()
            }
            for key, val in stats.items():
                st.metric(key, f"{val:.1f}")
            
        else:
            # For categorical data: show category counts
            cat_counts = map_data[map_type].value_counts().reset_index()
            cat_counts.columns = [map_type, 'Count']
            st.dataframe(cat_counts, use_container_width=True)
            
            st.markdown("---")
            st.markdown("**📊 States by Category:**")
            for category in sorted(map_data[map_type].unique()):
                states = map_data[map_data[map_type] == category]['State'].tolist()
                with st.expander(f"{category} ({len(states)} states)"):
                    st.write(", ".join(states))
